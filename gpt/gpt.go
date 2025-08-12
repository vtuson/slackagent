package gpt

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"

	gpt "github.com/ayush6624/go-chatgpt"
)

const (
	OPENAIURL     = "https://api.openai.com/v1/chat/completions"
	SYSTEMROLE    = "system"
	USERROLE      = "user"
	ASSISTANTROLE = "assistant"
	MODELGPT35    = "gpt-3.5-turbo"
)

type GPTmessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAI struct {
	apiKey string
	model  string
}

func GetDefaultModel() string {
	return MODELGPT35
}

func (o *OpenAI) SetApiKey(key string) {
	o.apiKey = key
}

func (o *OpenAI) SetModel(model string) {
	o.model = model
}

func (o *OpenAI) GptQuery(systemPrompt string, message string, context string) (string, error) {

	systemMessage := GPTmessage{
		Role:    SYSTEMROLE,
		Content: systemPrompt,
	}

	userMessage := GPTmessage{
		Role:    USERROLE,
		Content: message,
	}

	var messages []interface{}

	if context == "" {
		// Only include system and user messages if context is empty
		messages = []interface{}{systemMessage, userMessage}
	} else {
		// Include all three messages if context is provided
		contextMessage := GPTmessage{
			Role:    USERROLE,
			Content: context,
		}
		messages = []interface{}{systemMessage, userMessage, contextMessage}
	}
	data := map[string]interface{}{
		"model":    o.model,
		"messages": messages,
	}
	return o.gptSend(data)

}

func (o *OpenAI) gptSend(data map[string]interface{}) (string, error) {

	jsonData, _ := json.Marshal(data)

	req, err := http.NewRequest("POST", OPENAIURL, bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Println("Failed to create API request " + err.Error())
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	client := http.DefaultClient
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("Failed to make API request" + err.Error())
		return "", err
	}

	defer resp.Body.Close()
	respBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("Failed to read API response " + err.Error())
		return "", err
	}

	var gptMesg gpt.ChatResponse
	err = json.Unmarshal(respBody, &gptMesg)
	if err != nil {
		fmt.Printf("Error unmarshalling JSON: %v\n", err)
		return "", err
	}

	var reply string
	if len(gptMesg.Choices) > 0 {
		reply = gptMesg.Choices[0].Message.Content
	} else {
		log.Printf("%v\n", gptMesg)
		return "", errors.New("no reply")
	}

	return reply, nil

}
