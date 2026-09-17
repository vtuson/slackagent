package gpt

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

const (
	// MODELCLAUDE is the default Anthropic model.
	MODELCLAUDE = "claude-opus-5"
	// CLAUDEMAXTOKENS is the default response ceiling. Unlike OpenAI, the
	// Messages API requires max_tokens on every request.
	CLAUDEMAXTOKENS = 1024
)

// Claude talks to the Anthropic Messages API. It mirrors the OpenAI type so
// the two are interchangeable behind the LLM interface.
type Claude struct {
	apiKey    string
	model     string
	url       string
	maxTokens int64

	client *anthropic.Client
}

func (c *Claude) SetApiKey(key string) {
	c.apiKey = key
	c.client = nil
}

func (c *Claude) SetModel(model string) {
	c.model = model
}

// SetURL overrides the API base URL. Mainly useful for pointing tests at an
// httptest server or routing through a proxy.
func (c *Claude) SetURL(url string) {
	c.url = url
	c.client = nil
}

// SetMaxTokens overrides the default response ceiling.
func (c *Claude) SetMaxTokens(maxTokens int64) {
	c.maxTokens = maxTokens
}

// getClient builds the SDK client on first use, so the setters can be called
// in any order before the first query.
func (c *Claude) getClient() *anthropic.Client {
	if c.client != nil {
		return c.client
	}

	opts := []option.RequestOption{}
	if c.apiKey != "" {
		opts = append(opts, option.WithAPIKey(c.apiKey))
	}
	if c.url != "" {
		opts = append(opts, option.WithBaseURL(c.url))
	}

	client := anthropic.NewClient(opts...)
	c.client = &client
	return c.client
}

// GptQuery sends a single-turn query. The system prompt maps onto Anthropic's
// top-level system field rather than a message role, and any context is
// appended to the user turn.
func (c *Claude) GptQuery(systemPrompt string, message string, context string) (string, error) {
	userText := message
	if context != "" {
		userText = message + "\n\n" + context
	}

	model := c.model
	if model == "" {
		model = MODELCLAUDE
	}

	maxTokens := c.maxTokens
	if maxTokens <= 0 {
		maxTokens = CLAUDEMAXTOKENS
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: maxTokens,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(userText)),
		},
	}
	if systemPrompt != "" {
		params.System = []anthropic.TextBlockParam{{Text: systemPrompt}}
	}

	resp, err := c.getClient().Messages.New(ctxBackground(), params)
	if err != nil {
		return "", fmt.Errorf("anthropic request failed: %w", err)
	}

	if resp.StopReason == anthropic.StopReasonRefusal {
		return "", fmt.Errorf("anthropic declined the request: %s", resp.StopDetails.Explanation)
	}

	for _, block := range resp.Content {
		if text, ok := block.AsAny().(anthropic.TextBlock); ok {
			return text.Text, nil
		}
	}

	log.Printf("anthropic returned no text block (stop_reason=%s, blocks=%d)\n", resp.StopReason, len(resp.Content))
	return "", errors.New("no reply")
}

// ctxBackground is split out so callers that need cancellation can be added
// later without changing the LLM interface.
func ctxBackground() context.Context {
	return context.Background()
}
