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
	OPENAIURL      = "https://api.openai.com/v1/chat/completions"
	OPENAIEMBEDURL = "https://api.openai.com/v1/embeddings"
	SYSTEMROLE     = "system"
	USERROLE       = "user"
	ASSISTANTROLE  = "assistant"
	MODELGPT35     = "gpt-3.5-turbo"
	MODELEMBEDDING = "text-embedding-3-small"
)

type GPTmessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAI struct {
	apiKey string
	model  string
	url    string
}

// embeddingResponse represents the OpenAI embedding API response
type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index,omitempty"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func (o *OpenAI) SetURL(url string) {
	o.url = url
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

	url := o.url
	if url == "" {
		url = OPENAIURL
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
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

// GetEmbedding generates an embedding vector for the given text using OpenAI's embedding API
func (o *OpenAI) GetEmbedding(text string) ([]float32, error) {
	data := map[string]interface{}{
		"model": MODELEMBEDDING,
		"input": text,
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", OPENAIEMBEDURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create API request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	client := http.DefaultClient
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make API request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read API response: %w", err)
	}

	// Parse the embedding response
	var embeddingResp embeddingResponse

	err = json.Unmarshal(respBody, &embeddingResp)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if embeddingResp.Error != nil {
		return nil, fmt.Errorf("OpenAI API error: %s", embeddingResp.Error.Message)
	}

	if len(embeddingResp.Data) == 0 || len(embeddingResp.Data[0].Embedding) == 0 {
		return nil, errors.New("empty embedding response")
	}

	return embeddingResp.Data[0].Embedding, nil
}

// GetEmbeddingsBatch generates embedding vectors for multiple texts in a single API call
func (o *OpenAI) GetEmbeddingsBatch(texts []string) ([][]float32, error) {
	data := map[string]interface{}{
		"model": MODELEMBEDDING,
		"input": texts,
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", OPENAIEMBEDURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create API request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	client := http.DefaultClient
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make API request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read API response: %w", err)
	}

	// Parse the embedding response
	var embeddingResp embeddingResponse

	err = json.Unmarshal(respBody, &embeddingResp)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if embeddingResp.Error != nil {
		return nil, fmt.Errorf("OpenAI API error: %s", embeddingResp.Error.Message)
	}

	if len(embeddingResp.Data) == 0 {
		return nil, errors.New("empty embedding response")
	}

	// Sort by index to maintain order
	embeddings := make([][]float32, len(embeddingResp.Data))
	for _, item := range embeddingResp.Data {
		embeddings[item.Index] = item.Embedding
	}

	return embeddings, nil
}
