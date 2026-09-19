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
// the newer models reject the parameter outright rather than ignoring it.
func (c *Claude) SetTemperature(temperature *float64) {
	c.temperature = temperature
}

// SetEffort sets output_config.effort, which replaced the sampling params on
// the newer models.
func (c *Claude) SetEffort(effort string) {
	c.effort = effort
}

// modelOrDefault is the model this client will actually send, which is the
// default whenever none was configured.
func (c *Claude) modelOrDefault() string {
	if c.model == "" {
		return MODELCLAUDE
	}
	return c.model
}

// explainRequestError is the single exit for a failed Messages call. A knob
// the model no longer accepts surfaces here, as a 400, rather than at
// startup, so the wrapper names the knob before the error reaches a caller
// that has no idea a temperature was ever set.
func (c *Claude) explainRequestError(err error) error {
	return fmt.Errorf("anthropic request failed: %w", explainKnobRejection(err, c.modelOrDefault(), c.temperature, c.effort))
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

// baseParams builds the half of a request that does not depend on the
// conversation: model, ceiling and any configured knobs. Query and chat turns
// share it so a knob cannot end up applied to one and not the other.
func (c *Claude) baseParams() anthropic.MessageNewParams {
	maxTokens := c.maxTokens
	if maxTokens <= 0 {
		maxTokens = CLAUDEMAXTOKENS
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(c.modelOrDefault()),
		MaxTokens: maxTokens,
	}
	// Both knobs are left off the request unless configured, and go out as
	// given: whether this model still accepts them is the API's call, and a
	// refusal comes back through explainRequestError.
	if c.temperature != nil {
		params.Temperature = anthropic.Float(*c.temperature)
	}
	if c.effort != "" {
		params.OutputConfig = anthropic.OutputConfigParam{
			Effort: anthropic.OutputConfigEffort(c.effort),
		}
	}
	return params
}

// claudeSystem wraps a system prompt in the block list the API expects. An
// empty prompt gives nil, which leaves the field off the request.
func claudeSystem(systemPrompt string) []anthropic.TextBlockParam {
	if systemPrompt == "" {
		return nil
	}
	return []anthropic.TextBlockParam{{Text: systemPrompt}}
}

// claudeTools converts the neutral tool definitions into Anthropic's shape.
// The schema is passed through as the map it already is, rather than being
// rebuilt field by field, so a schema from an MCP server survives intact.
func claudeTools(tools []Tool) []anthropic.ToolUnionParam {
	if len(tools) == 0 {
		return nil
	}

	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		schema := anthropic.ToolInputSchemaParam{}
		if t.InputSchema != nil {
			if props, ok := t.InputSchema["properties"]; ok {
				schema.Properties = props
			}
			if req, ok := t.InputSchema["required"].([]string); ok {
				schema.Required = req
			} else if req, ok := t.InputSchema["required"].([]any); ok {
				// A schema decoded from JSON gives []any, not
				// []string.
				for _, r := range req {
					if name, ok := r.(string); ok {
						schema.Required = append(schema.Required, name)
					}
				}
			}
		}

		tool := anthropic.ToolParam{
			Name:        t.Name,
			InputSchema: schema,
		}
		if t.Description != "" {
			tool.Description = anthropic.String(t.Description)
		}
		out = append(out, anthropic.ToolUnionParam{OfTool: &tool})
	}
	return out
}

// claudeResponse converts an API message into the shared Response, pulling out
// the reply text and any tool the model is asking for.
func claudeResponse(msg *anthropic.Message) *Response {
	out := &Response{
		StopReason:    normaliseClaudeStop(msg.StopReason),
		RawStopReason: string(msg.StopReason),
	}
	if msg.StopReason == anthropic.StopReasonRefusal {
		out.Detail = msg.StopDetails.Explanation
	}

	for _, block := range msg.Content {
		switch variant := block.AsAny().(type) {
		case anthropic.TextBlock:
			// A tool-use turn can carry a preamble before the
			// call; keep the first block, as a plain reply only
			// ever has one.
			if out.Text == "" {
				out.Text = variant.Text
			}
		case anthropic.ToolUseBlock:
			out.ToolCalls = append(out.ToolCalls, ToolCall{
				ID:   variant.ID,
				Name: variant.Name,
				// Input is json.RawMessage on the block, but
				// the SDK only fills the raw JSON, so read it
				// from the JSON view.
				Input: []byte(variant.JSON.Input.Raw()),
			})
		}
	}

	return out
}

// Query sends a single-turn query and reports why generation stopped. The
// system prompt maps onto Anthropic's top-level system field rather than a
// message role, and any context is appended to the user turn.
//
// It offers no tools; a tool call needs a second request carrying the turn it
// was asked in, which only NewChat can do.
func (c *Claude) Query(systemPrompt string, message string, context string) (*Response, error) {
	userText := message
	if context != "" {
		userText = message + "\n\n" + context
	}

	params := c.baseParams()
	params.System = claudeSystem(systemPrompt)
	params.Messages = []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock(userText)),
	}

	resp, err := c.getClient().Messages.New(ctxBackground(), params)
	if err != nil {
		return nil, c.explainRequestError(err)
	}

	return claudeResponse(resp), nil
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

// claudeChat is a Claude conversation. History is kept in the SDK's own
// message type, so an assistant turn goes back to the API exactly as it came
// out, tool_use blocks and all.
type claudeChat struct {
	c        *Claude
	system   []anthropic.TextBlockParam
	tools    []anthropic.ToolUnionParam
	messages []anthropic.MessageParam
}

// NewChat starts a conversation with a fixed system prompt and tool set.
func (c *Claude) NewChat(systemPrompt string, tools []Tool) Chat {
	return &claudeChat{
		c:      c,
		system: claudeSystem(systemPrompt),
		tools:  claudeTools(tools),
	}
}

// Send adds a user message and returns the reply.
func (ch *claudeChat) Send(message string) (*Response, error) {
	ch.messages = append(ch.messages, anthropic.NewUserMessage(anthropic.NewTextBlock(message)))
	return ch.send()
}

// SendToolResults answers the outstanding calls. They go back as a single user
// turn: Anthropic rejects a turn that answers only some of them, and splitting
// them over several turns is the same thing as far as the API is concerned.
func (ch *claudeChat) SendToolResults(results []ToolResult) (*Response, error) {
	if len(results) == 0 {
		return nil, errors.New("no tool results to send")
	}

	blocks := make([]anthropic.ContentBlockParamUnion, 0, len(results))
	for _, r := range results {
		blocks = append(blocks, anthropic.NewToolResultBlock(r.ID, r.Content, r.IsError))
	}
	ch.messages = append(ch.messages, anthropic.NewUserMessage(blocks...))
	return ch.send()
}

// send issues the request and records the reply, so the next turn carries it.
func (ch *claudeChat) send() (*Response, error) {
	params := ch.c.baseParams()
	params.System = ch.system
	params.Tools = ch.tools
	params.Messages = ch.messages

	resp, err := ch.c.getClient().Messages.New(ctxBackground(), params)
	if err != nil {
		return nil, ch.c.explainRequestError(err)
	}

	// The assistant turn is appended before the tools run: the results are
	// only accepted alongside the turn that asked for them.
	ch.messages = append(ch.messages, resp.ToParam())

	return claudeResponse(resp), nil
}
