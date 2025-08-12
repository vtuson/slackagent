package slack

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/slack-go/slack"
	"github.com/slack-go/slack/slackevents"
	"github.com/slack-go/slack/socketmode"
)

type Client struct {
	api          *slack.Client
	socketClient *socketmode.Client
	channelID    string
	threadMax    int
}

func (c *Client) SetThreadMax(threadMax int) {
	c.threadMax = threadMax
}

// New creates a new Slack client with the given bot token and app-level token
func New(botToken string, appToken string, channelID string) *Client {
	// Add debug logging
	log.Printf("Initializing Slack client with bot token: %s...", botToken[:10]+"...")
	log.Printf("Using app token: %s...", appToken[:10]+"...")

	api := slack.New(botToken, slack.OptionAppLevelToken(appToken))

	// Test the bot token
	_, err := api.AuthTest()
	if err != nil {
		log.Printf("Warning: Bot token test failed: %v", err)
	} else {
		log.Println("Bot token test successful")
	}

	// Initialize socket mode with more options
	socketClient := socketmode.New(
		api,
		socketmode.OptionDebug(true),
		socketmode.OptionLog(log.New(os.Stdout, "socketmode: ", log.Lshortfile|log.LstdFlags)),
	)

	return &Client{
		api:          api,
		socketClient: socketClient,
		channelID:    channelID,
		threadMax:    20,
	}
}

// Start starts the Slack client and listens for events
func (c *Client) Start(Processor func(event interface{})) error {
	log.Println("Starting Slack client...")

	// Create a channel to handle OS signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Start the socket mode client
	go func() {
		for evt := range c.socketClient.Events {
			switch evt.Type {
			case socketmode.EventTypeConnecting:
				log.Println("Connecting to Slack with Socket Mode...")
			case socketmode.EventTypeConnectionError:
				log.Printf("Connection failed: %+v", evt.Data)
			case socketmode.EventTypeConnected:
				log.Println("Connected to Slack with Socket Mode.")
			case socketmode.EventTypeEventsAPI:
				eventsAPIEvent, ok := evt.Data.(slackevents.EventsAPIEvent)
				if !ok {
					log.Printf("Ignored %+v\n", evt)
					continue
				}
				log.Printf("I got an event: %v\n", eventsAPIEvent)

				c.socketClient.Ack(*evt.Request)

				switch eventsAPIEvent.Type {
				case slackevents.CallbackEvent:
					innerEvent := eventsAPIEvent.InnerEvent
					Processor(innerEvent.Data)

				default:
					c.socketClient.Debugf("unsupported Events API event received")
				}
			}
		}
	}()

	// Start the socket mode client
	log.Println("Running socket mode client...")
	err := c.socketClient.Run()
	if err != nil {
		return fmt.Errorf("error running socket mode client: %v", err)
	}

	// Wait for OS signal
	<-sigChan
	return nil
}

// SendMessage sends a message to the configured Slack channel
func (c *Client) SendMessage(msgText string) error {
	_, _, err := c.api.PostMessage(
		c.channelID,
		slack.MsgOptionText(msgText, false),
	)
	if err != nil {
		return fmt.Errorf("error sending message: %v", err)
	}

	fmt.Println("Message sent successfully!")
	return nil
}

// SendTestMessage sends a test message and starts the client, returns true if program should exit
func (c *Client) SendTestMessage(testMessage string) bool {
	if testMessage == "" {
		return false // Continue execution
	}

	if _, err := c.PostInChannel(c.channelID, testMessage); err != nil {
		log.Fatalf("Failed to send test message to Slack: %v", err)
	}
	fmt.Println("Test message sent successfully!")
	return true // Exit program
}

// returns the timestamp of the message and the error if any
func (c *Client) PostInChannel(channel string, message string) (string, error) {
	_, ts, err := c.api.PostMessage(
		channel,
		slack.MsgOptionText(message, false),
	)
	if err != nil {
		return "", fmt.Errorf("error sending message: %v", err)
	}
	return ts, nil
}

// returns the timestamp of the message and the error if any
func (c *Client) PostInThread(channel string, message string, threadTimeStamp string) (string, error) {
	_, ts, err := c.api.PostMessage(
		channel,
		slack.MsgOptionText(message, false),
		slack.MsgOptionTS(threadTimeStamp), // This makes it a thread reply
	)
	if err != nil {
		return "", fmt.Errorf("error sending message: %v", err)
	}
	return ts, nil
}

// AddText formats text with optional bold formatting
func AddText(newText string, to string, bold bool) string {
	if bold {
		return to + "\n*" + newText + "*"
	}
	return to + "\n" + newText
}

func (c *Client) GetThreadMessages(channelID string, threadTS string) ([]slack.Message, error) {
	replies, _, _, err := c.api.GetConversationRepliesContext(
		context.Background(),
		&slack.GetConversationRepliesParameters{
			ChannelID: channelID,
			Timestamp: threadTS,
			Limit:     c.threadMax,
		},
	)
	if err != nil {
		return nil, err
	}

	return replies, nil
}

func StripAtMention(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return text
	}

	if strings.HasPrefix(text, "@") {
		// Find the first space to identify the end of the first word
		spaceIndex := strings.Index(text, " ")
		if spaceIndex == -1 {
			// If no space found, the entire text is just the @ mention
			return ""
		}
		// Return everything after the first space, trimmed
		return strings.TrimSpace(text[spaceIndex+1:])
	}
	return text
}
