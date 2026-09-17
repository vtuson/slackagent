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
	apiKey    string
	model     string
	url       string
	maxTokens int64
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

// SetMaxTokens caps the reply length. Unlike Anthropic, OpenAI treats
// max_tokens as optional, so zero or less leaves it out of the request.
func (o *OpenAI) SetMaxTokens(maxTokens int64) {
	o.maxTokens = maxTokens
}

// normaliseOpenAIStop maps OpenAI finish reasons onto the shared STOP* set.
func normaliseOpenAIStop(reason string) string {
	switch reason {
	case "stop":
		return STOPEND
	case "length":
		return STOPMAXTOKENS
	case "tool_calls", "function_call":
		return STOPTOOLUSE
	case "content_filter":
		return STOPREFUSAL
	default:
		return STOPOTHER
	}
}

// Query sends a single-turn query and reports why generation stopped.
func (o *OpenAI) Query(systemPrompt string, message string, context string) (*Response, error) {

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
	if o.maxTokens > 0 {
		data["max_tokens"] = o.maxTokens
	}
	return o.gptSend(data)
}

// GptQuery returns just the reply text. Callers that need to react to the stop
// reason should use Query instead.
func (o *OpenAI) GptQuery(systemPrompt string, message string, context string) (string, error) {
	resp, err := o.Query(systemPrompt, message, context)
	if err != nil {
		return "", err
	}
	if resp.Text == "" {
		return "", errors.New("no reply")
	}
	return resp.Text, nil
}

func (o *OpenAI) gptSend(data map[string]interface{}) (*Response, error) {

	jsonData, _ := json.Marshal(data)

	url := o.url
	if url == "" {
		url = OPENAIURL
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Println("Failed to create API request " + err.Error())
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	client := http.DefaultClient
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("Failed to make API request" + err.Error())
		return nil, err
	}

	defer resp.Body.Close()
	respBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("Failed to read API response " + err.Error())
		return nil, err
	}

	var gptMesg gpt.ChatResponse
	err = json.Unmarshal(respBody, &gptMesg)
	if err != nil {
		fmt.Printf("Error unmarshalling JSON: %v\n", err)
		return nil, err
	}

	if len(gptMesg.Choices) == 0 {
		log.Printf("openai returned no choices\n")
		return &Response{StopReason: STOPOTHER}, nil
	}

	choice := gptMesg.Choices[0]
	return &Response{
		Text:          choice.Message.Content,
		StopReason:    normaliseOpenAIStop(choice.FinishReason),
		RawStopReason: choice.FinishReason,
	}, nil
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
