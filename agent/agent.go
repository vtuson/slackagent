package agent

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/slack-go/slack/slackevents"
	"github.com/vtuson/slackagent/gpt"
	"github.com/vtuson/slackagent/mail"
	"github.com/vtuson/slackagent/slack"

	"gopkg.in/yaml.v2"
)

// Config represents the YAML configuration structure
type Config struct {
	Slack *struct {
		Token    string `yaml:"token,omitempty"`
		AppToken string `yaml:"app_token,omitempty"`
		Channel  string `yaml:"channel"`
	} `yaml:"slack"`
	GPT *struct {
		Key   string `yaml:"key"`
		Model string `yaml:"model"`
	} `yaml:"gpt"`
	Mail *struct {
		Label     string `yaml:"label"`
		MaxID     string `yaml:"maxid,omitempty"`
		Wait      int    `yaml:"wait,omitempty"`
		Secret    string `yaml:"secret"`
		AuthToken string `yaml:"auth_token,omitempty"`
	} `yaml:"mail"`
	AgentConfig interface{} `yaml:"agent_config"`
}

type Agent struct {
	slackClient    *slack.Client
	gptApiKey      string
	Config         *Config
	test           bool
	EmailProcessor func(email mail.Email)
	SlackProcessor func(event interface{})
	MCPClient      *MCPClient
}

func (a *Agent) GetCustomConfig(customConfig interface{}) error {
	data, err := yaml.Marshal(a.Config.AgentConfig.(map[interface{}]interface{}))
	if err != nil {
		return err
	}
	return yaml.Unmarshal(data, customConfig)
}

func (a *Agent) GetSlackClient() *slack.Client {
	return a.slackClient
}

// loadConfig reads and parses the YAML configuration file
func (a *Agent) LoadConfig(configPath string) error {
	data, err := ioutil.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config file: %v", err)
	}

	var config Config
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return fmt.Errorf("failed to parse config file: %v", err)
	}

	// Validate config values
	if config.Slack == nil {
		log.Fatal("Slack configuration is required in config file")
	}

	// Validate config values
	if config.Slack.Token == "" {
		log.Fatal("Slack bot token is required in config file")
	}

	if config.Slack.AppToken == "" {
		log.Fatal("Slack app-level token is required for Socket Mode in config file")
	}

	if config.Slack.Channel == "" {
		log.Fatal("Slack channel ID is required in config file")
	}

	if config.GPT == nil {
		log.Println("GPT configuration is not required in config file")
	} else if config.GPT.Key == "" {
		log.Fatal("OpenAI API Key is required in config file")
	} else {
		a.gptApiKey = config.GPT.Key
	}

	a.Config = &config
	return nil
}

// initializeSlackClient creates and starts the Slack client
func (a *Agent) InitializeSlackClient() {
	//create slack client by default
	client := slack.New(a.Config.Slack.Token, a.Config.Slack.AppToken, a.Config.Slack.Channel)

	// Start the Slack client in a goroutine
	go func() {
		if err := client.Start(a.slackFilter); err != nil {
			log.Fatalf("Error running Slack client: %v", err)
		}
	}()

	a.slackClient = client
}

func (a *Agent) HasEmail() bool {
	return a.Config.Mail != nil
}

func (a *Agent) SetTestMode(test bool) {
	a.test = test
}

// processEmails handles the email processing logic
func (a *Agent) ProcessEmails() {
	if !a.HasEmail() {
		log.Println("no email conf provided")
		return
	}
	labelToParse := a.Config.Mail.Label
	if a.test {
		labelToParse = "test"
		log.Println("test mode is ON")
	} else {
		log.Println("test mode is OFF")
	}

	srv, err := mail.Connect(a.Config.Mail.Secret, a.Config.Mail.AuthToken)
	if err != nil {
		return
	}
	label, err := mail.GetIdForLabel(srv, labelToParse)
	if err != nil {
		log.Println(err.Error())
		return
	}

	nextID := a.Config.Mail.MaxID
	if nextID == "" {
		nextID = mail.NO_MAX_ID
	}
	err = nil
	for {
		err, nextID = mail.GetEmails(srv, nextID, label, a.EmailProcessor)
		if err != nil {
			nextID = mail.NO_MAX_ID
			log.Println("did not find any msgs")
		}
		// Sleep for 15 minutes (900 seconds)
		durationSleep := 15
		if a.Config.Mail.Wait > 0 {
			durationSleep = a.Config.Mail.Wait
		}
		time.Sleep(time.Duration(durationSleep) * time.Minute)
	}
}
func (a *Agent) NewLLM() *gpt.OpenAI {
	var openai gpt.OpenAI
	openai.SetApiKey(a.Config.GPT.Key)
	if a.Config.GPT.Model == "" {
		a.Config.GPT.Model = gpt.GetDefaultModel()
		log.Println("Using default model: ", a.Config.GPT.Model)
	}
	openai.SetModel(a.Config.GPT.Model)

	return &openai
}

func (a *Agent) slackFilter(event interface{}) {

	switch ev := event.(type) {
	case *slackevents.AppMentionEvent:
		if ev.BotID != "" {
			log.Println("Bot message, skipping")
			return
		}
		a.SlackProcessor(event)

	case *slackevents.MessageEvent:
		if ev.BotID != "" {
			log.Println("Bot message, skipping")
			return
		}
		a.SlackProcessor(event)
	}
}

func (a *Agent) WaitForSignal() {
	sigs := make(chan os.Signal, 1)

	// Notify the channel on SIGINT and SIGTERM.
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	log.Println("Waiting for OS signal (Ctrl+C to stop)...")
	// Block until a signal is received.
	<-sigs
	log.Println("Signal received, exiting.")
}
