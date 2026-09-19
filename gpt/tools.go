package gpt

import (
	"encoding/json"
	"errors"
	"fmt"
)

// DEFAULTTOOLTURNS caps RunToolLoop when the caller does not pick a limit. A
// model that keeps asking for tools forever would otherwise bill for it
// forever.
const DEFAULTTOOLTURNS = 10

// Tool is a provider-neutral tool definition. Both providers take the same
// three things, only wrapped differently on the wire: Anthropic nests the
// schema under input_schema, OpenAI under function.parameters.
type Tool struct {
	Name string
	// Description is what the model reads to decide whether to call the
	// tool, so it does more work than the name. Be generous with it.
	Description string
	// InputSchema is a JSON Schema object describing the arguments, the
	// same shape an MCP server publishes. A nil schema is sent as an object
	// with no properties, which is the right thing for a tool that takes no
	// arguments.
	InputSchema map[string]any
}

// ToolCall is the model asking for a tool to be run. It arrives on a Response
// whose StopReason is STOPTOOLUSE.
type ToolCall struct {
	// ID identifies this call and must be echoed back on the matching
	// ToolResult. Providers reject a result that names no call.
	ID   string
	Name string
	// Input is the raw JSON arguments object. It is left raw because the
	// model, not the schema, decides what actually turns up in it; decode
	// it with Arguments or straight into a struct of your own.
	Input json.RawMessage
}

// Arguments decodes the call arguments into a map. Callers with a known shape
// should json.Unmarshal Input into their own struct instead.
func (t ToolCall) Arguments() (map[string]any, error) {
	if len(t.Input) == 0 {
		return map[string]any{}, nil
	}
	var args map[string]any
	if err := json.Unmarshal(t.Input, &args); err != nil {
		return nil, fmt.Errorf("decoding arguments for tool %q: %w", t.Name, err)
	}
	return args, nil
}

// ToolResult is what the tool produced, on its way back to the model.
type ToolResult struct {
	// ID is the ToolCall.ID this answers.
	ID      string
	Content string
	// IsError marks a failed call. Report failures this way rather than as
	// a Go error: the model can read the message and try something else,
	// which it cannot do if the loop aborts. Anthropic has a flag for it;
	// on OpenAI the content is prefixed instead, since the API has no
	// equivalent field.
	IsError bool
}

// ToolFunc runs one tool call. It returns a ToolResult rather than an error
// because a failed tool is something the model should see, not something that
// should stop the conversation.
type ToolFunc func(call ToolCall) ToolResult

// Chat is a multi-turn conversation with a fixed system prompt and tool set.
//
// It exists because a tool call cannot be answered within a single request:
// the model asks, and the answer only reaches it on a second request that also
// carries the turn it asked in. Each provider keeps that history in its own
// native format, so nothing is converted between them and nothing is lost in
// the round trip. A Chat is not safe for concurrent use; give each
// conversation its own.
type Chat interface {
	// Send adds a user message and returns the reply. A reply whose
	// StopReason is STOPTOOLUSE carries the requested calls in ToolCalls.
	Send(message string) (*Response, error)
	// SendToolResults answers the outstanding calls and returns the next
	// reply. Every call in the previous response must be answered, in one
	// go: both providers reject a turn that leaves one unanswered.
	SendToolResults(results []ToolResult) (*Response, error)
}

// RunToolLoop sends a message and keeps running tools until the model stops
// asking for them, which is the usual way to drive a Chat. maxTurns bounds the
// number of tool rounds; zero means DEFAULTTOOLTURNS.
//
// Callers that need to inspect or approve individual calls should drive the
// Chat directly instead.
func RunToolLoop(chat Chat, message string, run ToolFunc, maxTurns int) (*Response, error) {
	if run == nil {
		return nil, errors.New("no tool runner provided")
	}
	if maxTurns <= 0 {
		maxTurns = DEFAULTTOOLTURNS
	}

	resp, err := chat.Send(message)
	if err != nil {
		return nil, err
	}

	for turns := 0; resp.StopReason == STOPTOOLUSE; turns++ {
		if turns >= maxTurns {
			return resp, fmt.Errorf("tool loop still running after %d turns, giving up", maxTurns)
		}
		if len(resp.ToolCalls) == 0 {
			// The stop reason says tool use but no call came with it.
			// Answering nothing would loop forever, so stop here.
			return resp, errors.New("model stopped for tool use but requested no tools")
		}

		results := make([]ToolResult, 0, len(resp.ToolCalls))
		for _, call := range resp.ToolCalls {
			results = append(results, run(call))
		}

		resp, err = chat.SendToolResults(results)
		if err != nil {
			return nil, err
		}
	}

	return resp, nil
}
