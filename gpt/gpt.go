package gpt

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
)

const (
	OPENAIURL      = "https://api.openai.com/v1/chat/completions"
	OPENAIEMBEDURL = "https://api.openai.com/v1/embeddings"
	SYSTEMROLE     = "system"
	USERROLE       = "user"
	ASSISTANTROLE  = "assistant"
	// TOOLROLE carries a tool result back to the model. Anthropic sends
	// the same thing as a block inside a user turn.
	TOOLROLE       = "tool"
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

// openaiResponse is the slice of the chat completions response this package
// reads. It replaces the go-chatgpt type, which has no tool_calls field and so
// decoded a tool call as an empty reply.
type openaiResponse struct {
	Choices []struct {
		// Message stays raw so the assistant turn can go back into the
		// conversation byte for byte, tool calls and all.
		Message      json.RawMessage `json:"message"`
		FinishReason string          `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// openaiMessage is an assistant turn, decoded from the raw message above.
type openaiMessage struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	ToolCalls []struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
			// Arguments is a JSON object encoded as a string,
			// which is OpenAI's shape, not Anthropic's.
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
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
	resp, err := o.post(data)
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		log.Printf("openai returned no choices\n")
		return &Response{StopReason: STOPOTHER}, nil
	}

	choice := resp.Choices[0]
	var msg openaiMessage
	if err := json.Unmarshal(choice.Message, &msg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal assistant message: %w", err)
	}

	out := &Response{
		Text:          msg.Content,
		StopReason:    normaliseOpenAIStop(choice.FinishReason),
		RawStopReason: choice.FinishReason,
	}
	for _, call := range msg.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, ToolCall{
			ID:   call.ID,
			Name: call.Function.Name,
			// Arguments is JSON inside a string here, so the bytes
			// of that string are the raw argument object.
			Input: []byte(call.Function.Arguments),
		})
	}

	return out, nil
}

// post sends a chat completions request and decodes the envelope. It is split
// out from gptSend so a chat turn can read the raw assistant message and put
// it back in the conversation unchanged.
func (o *OpenAI) post(data map[string]interface{}) (*openaiResponse, error) {

	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

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

	var decoded openaiResponse
	if err := json.Unmarshal(respBody, &decoded); err != nil {
		fmt.Printf("Error unmarshalling JSON: %v\n", err)
		return nil, err
	}

	if decoded.Error != nil {
		return nil, fmt.Errorf("OpenAI API error: %s", decoded.Error.Message)
	}

	return &decoded, nil
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

// openaiTools converts the neutral tool definitions into OpenAI's shape, which
// nests everything under a function object rather than putting it at the top
// level the way Anthropic does.
func openaiTools(tools []Tool) []map[string]interface{} {
	if len(tools) == 0 {
		return nil
	}

	out := make([]map[string]interface{}, 0, len(tools))
	for _, t := range tools {
		schema := t.InputSchema
		if schema == nil {
			// A tool with no arguments still needs a schema; an
			// object with no properties is the way to say that.
			schema = map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
		}
		fn := map[string]interface{}{
			"name":       t.Name,
			"parameters": schema,
		}
		if t.Description != "" {
			fn["description"] = t.Description
		}
		out = append(out, map[string]interface{}{
			"type":     "function",
			"function": fn,
		})
	}
	return out
}

// openaiChat is an OpenAI conversation. Assistant turns are kept as the raw
// JSON they arrived as, so a tool call goes back to the API exactly as it came
// out rather than being rebuilt from parsed parts.
type openaiChat struct {
	o     *OpenAI
	tools []map[string]interface{}
	// messages holds map[string]interface{} for the turns this code writes
	// and json.RawMessage for the assistant turns the API wrote.
	messages []interface{}
}

// NewChat starts a conversation with a fixed system prompt and tool set.
func (o *OpenAI) NewChat(systemPrompt string, tools []Tool) Chat {
	ch := &openaiChat{o: o, tools: openaiTools(tools)}
	if systemPrompt != "" {
		ch.messages = append(ch.messages, GPTmessage{Role: SYSTEMROLE, Content: systemPrompt})
	}
	return ch
}

// Send adds a user message and returns the reply.
func (ch *openaiChat) Send(message string) (*Response, error) {
	ch.messages = append(ch.messages, GPTmessage{Role: USERROLE, Content: message})
	return ch.send()
}

// SendToolResults answers the outstanding calls. Each result is its own tool
// message, unlike Anthropic where they share one user turn, but they still all
// have to go in the same request: OpenAI rejects the next completion if a call
// is left unanswered.
func (ch *openaiChat) SendToolResults(results []ToolResult) (*Response, error) {
	if len(results) == 0 {
		return nil, errors.New("no tool results to send")
	}

	for _, r := range results {
		content := r.Content
		if r.IsError {
			// OpenAI has no is_error flag, so the only way to tell
			// the model the call failed is in the content.
			content = "Error: " + content
		}
		ch.messages = append(ch.messages, map[string]interface{}{
			"role":         TOOLROLE,
			"tool_call_id": r.ID,
			"content":      content,
		})
	}

	return ch.send()
}

// send issues the request and records the reply, so the next turn carries it.
func (ch *openaiChat) send() (*Response, error) {
	data := map[string]interface{}{
		"model":    ch.o.model,
		"messages": ch.messages,
	}
	if len(ch.tools) > 0 {
		data["tools"] = ch.tools
	}
	if ch.o.maxTokens > 0 {
		data["max_tokens"] = ch.o.maxTokens
	}
	if ch.o.temperature != nil {
		data["temperature"] = *ch.o.temperature
	}
	if ch.o.effort != "" {
		data["reasoning_effort"] = ch.o.effort
	}

	resp, err := ch.o.post(data)
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		log.Printf("openai returned no choices\n")
		return &Response{StopReason: STOPOTHER}, nil
	}

	choice := resp.Choices[0]
	// The assistant turn is appended before the tools run: the results are
	// only accepted alongside the turn that asked for them.
	ch.messages = append(ch.messages, choice.Message)

	var msg openaiMessage
	if err := json.Unmarshal(choice.Message, &msg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal assistant message: %w", err)
	}

	out := &Response{
		Text:          msg.Content,
		StopReason:    normaliseOpenAIStop(choice.FinishReason),
		RawStopReason: choice.FinishReason,
	}
	for _, call := range msg.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, ToolCall{
			ID:    call.ID,
			Name:  call.Function.Name,
			Input: []byte(call.Function.Arguments),
		})
	}

	return out, nil
}
