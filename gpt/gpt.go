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

// openaiReasoning lists the model families that reject temperature and take
// reasoning_effort instead. Everything else is the other way round: the chat
// models take temperature and have no effort dial.
//
// Prefixes, so point releases are covered. This table goes stale every time
// OpenAI ships a family; it is the only place to edit when they do.
var openaiReasoning = []string{"o1", "o3", "o4", "gpt-5"}

type GPTmessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAI struct {
	apiKey    string
	model     string
	url       string
	maxTokens int64

	// temperature is nil when unset. Zero is a real value here (it is the
	// usual request for determinism), so it cannot double as "unset" the
	// way maxTokens does.
	temperature *float64
	effort      string
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

// SetTemperature sets the sampling temperature. OpenAI's range is 0-2 with a
// default of 1, so a value ported straight from an Anthropic config means
// something different here. The reasoning models reject it; see Supports.
func (o *OpenAI) SetTemperature(temperature *float64) {
	o.temperature = temperature
}

// SetEffort sets reasoning_effort, which the reasoning models take in place of
// the sampling params.
func (o *OpenAI) SetEffort(effort string) {
	o.effort = effort
}

// Supports reports whether the configured model accepts a knob.
func (o *OpenAI) Supports(knob string) bool {
	model := o.model
	if model == "" {
		model = MODELGPT35
	}

	switch knob {
	case KNOBTEMPERATURE:
		return !hasAnyPrefix(model, openaiReasoning)
	case KNOBEFFORT:
		return hasAnyPrefix(model, openaiReasoning)
	default:
		return false
	}
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
	// Both knobs stay off the request unless configured. ApplyKnobs has
	// already refused any the model rejects, so no capability check here.
	if o.temperature != nil {
		data["temperature"] = *o.temperature
	}
	if o.effort != "" {
		data["reasoning_effort"] = o.effort
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
