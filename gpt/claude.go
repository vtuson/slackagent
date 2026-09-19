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

// claudeNoSampling lists the models that removed temperature/top_p/top_k.
// Sampling params were dropped on Opus 4.7 and on everything released after
// it; anything older still accepts them. Prefixes, so point releases are
// covered without listing each one.
//
// This table goes stale every time Anthropic ships a model. When a new model
// rejects a knob this is the only place to edit.
var claudeNoSampling = []string{
	"claude-opus-4-7",
	"claude-opus-4-8",
	"claude-opus-5",
	"claude-sonnet-5",
	"claude-fable-5",
	"claude-mythos-5",
}

// claudeEffort lists the models that accept output_config.effort. It arrived
// with Opus 4.5; Sonnet 4.5, Haiku 4.5 and the claude-3 family predate it.
var claudeEffort = []string{
	"claude-opus-4-5",
	"claude-opus-4-6",
	"claude-opus-4-7",
	"claude-opus-4-8",
	"claude-opus-5",
	"claude-sonnet-4-6",
	"claude-sonnet-5",
	"claude-fable-5",
	"claude-mythos-5",
}

// Claude talks to the Anthropic Messages API. It mirrors the OpenAI type so
// the two are interchangeable behind the LLM interface.
type Claude struct {
	apiKey    string
	model     string
	url       string
	maxTokens int64

	// temperature is nil when unset. It cannot use the "0 means unset"
	// shortcut maxTokens uses, because 0 is the value callers most often
	// want.
	temperature *float64
	effort      string

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

// SetTemperature sets the sampling temperature. Anthropic's range is 0-1, and
// the newer models reject the parameter outright; see Supports.
func (c *Claude) SetTemperature(temperature *float64) {
	c.temperature = temperature
}

// SetEffort sets output_config.effort, which replaced the sampling params on
// the newer models.
func (c *Claude) SetEffort(effort string) {
	c.effort = effort
}

// Supports reports whether the configured model accepts a knob.
func (c *Claude) Supports(knob string) bool {
	model := c.model
	if model == "" {
		model = MODELCLAUDE
	}

	switch knob {
	case KNOBTEMPERATURE:
		return !hasAnyPrefix(model, claudeNoSampling)
	case KNOBEFFORT:
		return hasAnyPrefix(model, claudeEffort)
	default:
		return false
	}
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

// normaliseClaudeStop maps Anthropic stop reasons onto the shared STOP* set.
func normaliseClaudeStop(reason anthropic.StopReason) string {
	switch reason {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence:
		return STOPEND
	case anthropic.StopReasonMaxTokens:
		return STOPMAXTOKENS
	case anthropic.StopReasonToolUse:
		return STOPTOOLUSE
	case anthropic.StopReasonRefusal:
		return STOPREFUSAL
	default:
		return STOPOTHER
	}
}

// Query sends a single-turn query and reports why generation stopped. The
// system prompt maps onto Anthropic's top-level system field rather than a
// message role, and any context is appended to the user turn.
func (c *Claude) Query(systemPrompt string, message string, context string) (*Response, error) {
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
	// Both knobs are left off the request unless configured. ApplyKnobs has
	// already refused any the model rejects, so no capability check here.
	if c.temperature != nil {
		params.Temperature = anthropic.Float(*c.temperature)
	}
	if c.effort != "" {
		params.OutputConfig = anthropic.OutputConfigParam{
			Effort: anthropic.OutputConfigEffort(c.effort),
		}
	}

	resp, err := c.getClient().Messages.New(ctxBackground(), params)
	if err != nil {
		return nil, fmt.Errorf("anthropic request failed: %w", err)
	}

	out := &Response{
		StopReason:    normaliseClaudeStop(resp.StopReason),
		RawStopReason: string(resp.StopReason),
	}
	if resp.StopReason == anthropic.StopReasonRefusal {
		out.Detail = resp.StopDetails.Explanation
	}

	for _, block := range resp.Content {
		if text, ok := block.AsAny().(anthropic.TextBlock); ok {
			out.Text = text.Text
			break
		}
	}

	return out, nil
}

// GptQuery returns just the reply text. Callers that need to react to the stop
// reason should use Query instead.
func (c *Claude) GptQuery(systemPrompt string, message string, context string) (string, error) {
	resp, err := c.Query(systemPrompt, message, context)
	if err != nil {
		return "", err
	}

	if resp.StopReason == STOPREFUSAL {
		return "", fmt.Errorf("anthropic declined the request: %s", resp.Detail)
	}

	if resp.Text == "" {
		log.Printf("anthropic returned no text (stop_reason=%s)\n", resp.RawStopReason)
		return "", errors.New("no reply")
	}

	return resp.Text, nil
}

// ctxBackground is split out so callers that need cancellation can be added
// later without changing the LLM interface.
func ctxBackground() context.Context {
	return context.Background()
}
