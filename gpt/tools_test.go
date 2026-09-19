package gpt

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// newSequenceServer returns a server that answers with each body in turn, and
// a pointer to the decoded requests it received. A tool loop is at least two
// round trips, so the tests need to script both sides of it.
func newSequenceServer(t *testing.T, responses ...string) (*httptest.Server, *[]map[string]interface{}) {
	t.Helper()

	var requests []map[string]interface{}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading request body: %v", err)
		}
		var body map[string]interface{}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("unmarshalling request body: %v", err)
		}
		requests = append(requests, body)

		if len(requests) > len(responses) {
			t.Errorf("unexpected request %d, only %d scripted", len(requests), len(responses))
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(responses[len(requests)-1]))
	}))
	t.Cleanup(srv.Close)

	return srv, &requests
}

// pageTool is a tool definition with the schema shape an MCP server publishes:
// required arrives as []any because it has been through JSON.
func pageTool() Tool {
	return Tool{
		Name:        "get_page",
		Description: "Fetch a page by name",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"page": map[string]any{"type": "string"}},
			"required":   []any{"page"},
		},
	}
}

const claudeToolUse = `{
	"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
	"content": [
		{"type": "text", "text": "let me look"},
		{"type": "tool_use", "id": "toolu_1", "name": "get_page", "input": {"page": "roadmap"}}
	],
	"stop_reason": "tool_use",
	"usage": {"input_tokens": 10, "output_tokens": 5}
}`

const claudeDone = `{
	"id": "msg_2", "type": "message", "role": "assistant", "model": "claude-opus-5",
	"content": [{"type": "text", "text": "the roadmap says ship it"}],
	"stop_reason": "end_turn",
	"usage": {"input_tokens": 20, "output_tokens": 5}
}`

const openaiToolUse = `{
	"id": "chat_1", "object": "chat.completion",
	"choices": [{
		"index": 0,
		"message": {"role": "assistant", "content": null, "tool_calls": [
			{"id": "call_1", "type": "function", "function": {"name": "get_page", "arguments": "{\"page\":\"roadmap\"}"}}
		]},
		"finish_reason": "tool_calls"
	}]
}`

const openaiDone = `{
	"id": "chat_2", "object": "chat.completion",
	"choices": [{
		"index": 0,
		"message": {"role": "assistant", "content": "the roadmap says ship it"},
		"finish_reason": "stop"
	}]
}`

func newClaudeAt(url string) *Claude {
	var c Claude
	c.SetApiKey("test-key")
	c.SetModel("claude-opus-5")
	c.SetURL(url)
	return &c
}

func newOpenAIAt(url string) *OpenAI {
	var o OpenAI
	o.SetApiKey("test-key")
	o.SetModel("gpt-3.5-turbo")
	o.SetURL(url)
	return &o
}

// The tool definition has to reach the wire, or the model has nothing to call.
func TestClaudeChatSendsToolDefinitions(t *testing.T) {
	srv, requests := newSequenceServer(t, claudeDone)
	chat := newClaudeAt(srv.URL).NewChat("be brief", []Tool{pageTool()})

	if _, err := chat.Send("what does the roadmap say?"); err != nil {
		t.Fatalf("Send returned error: %v", err)
	}

	tools, ok := (*requests)[0]["tools"].([]interface{})
	if !ok || len(tools) != 1 {
		t.Fatalf("expected one tool on the request, got %#v", (*requests)[0]["tools"])
	}
	tool := tools[0].(map[string]interface{})
	if tool["name"] != "get_page" {
		t.Errorf("tool name = %v, want get_page", tool["name"])
	}
	if tool["description"] != "Fetch a page by name" {
		t.Errorf("tool description = %v", tool["description"])
	}

	// Anthropic nests the schema under input_schema, and the required list
	// has to survive the []any it arrives as.
	schema, ok := tool["input_schema"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected an input_schema object, got %#v", tool["input_schema"])
	}
	if _, ok := schema["properties"].(map[string]interface{})["page"]; !ok {
		t.Errorf("page property missing from schema: %#v", schema)
	}
	required, ok := schema["required"].([]interface{})
	if !ok || len(required) != 1 || required[0] != "page" {
		t.Errorf("required = %#v, want [page]", schema["required"])
	}
}

// A tool_use reply has to come back as a call the caller can actually run:
// stop reason, id, name and arguments.
func TestClaudeChatReportsToolCalls(t *testing.T) {
	srv, _ := newSequenceServer(t, claudeToolUse)
	chat := newClaudeAt(srv.URL).NewChat("", []Tool{pageTool()})

	resp, err := chat.Send("what does the roadmap say?")
	if err != nil {
		t.Fatalf("Send returned error: %v", err)
	}

	if resp.StopReason != STOPTOOLUSE {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, STOPTOOLUSE)
	}
	if !resp.WantsTool() {
		t.Error("WantsTool() = false on a tool_use reply")
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(resp.ToolCalls))
	}
	call := resp.ToolCalls[0]
	if call.ID != "toolu_1" || call.Name != "get_page" {
		t.Errorf("call = %+v, want id toolu_1 name get_page", call)
	}
	args, err := call.Arguments()
	if err != nil {
		t.Fatalf("Arguments returned error: %v", err)
	}
	if args["page"] != "roadmap" {
		t.Errorf("args = %#v, want page roadmap", args)
	}
	// The preamble text comes back too, but it is not an answer.
	if resp.Text != "let me look" {
		t.Errorf("Text = %q, want %q", resp.Text, "let me look")
	}
}

// The second request must carry the assistant turn that asked for the tool as
// well as the result; Anthropic rejects a result that arrives on its own.
func TestClaudeChatSendsToolResultWithAskingTurn(t *testing.T) {
	srv, requests := newSequenceServer(t, claudeToolUse, claudeDone)
	chat := newClaudeAt(srv.URL).NewChat("", []Tool{pageTool()})

	resp, err := chat.Send("what does the roadmap say?")
	if err != nil {
		t.Fatalf("Send returned error: %v", err)
	}

	resp, err = chat.SendToolResults([]ToolResult{{ID: resp.ToolCalls[0].ID, Content: "ship it"}})
	if err != nil {
		t.Fatalf("SendToolResults returned error: %v", err)
	}
	if resp.Text != "the roadmap says ship it" {
		t.Errorf("Text = %q", resp.Text)
	}

	messages := (*requests)[1]["messages"].([]interface{})
	if len(messages) != 3 {
		t.Fatalf("got %d messages on the second request, want 3", len(messages))
	}

	assistant := messages[1].(map[string]interface{})
	if assistant["role"] != ASSISTANTROLE {
		t.Errorf("second message role = %v, want assistant", assistant["role"])
	}
	blocks := assistant["content"].([]interface{})
	var sawToolUse bool
	for _, b := range blocks {
		if b.(map[string]interface{})["type"] == "tool_use" {
			sawToolUse = true
		}
	}
	if !sawToolUse {
		t.Errorf("the replayed assistant turn lost its tool_use block: %#v", blocks)
	}

	result := messages[2].(map[string]interface{})["content"].([]interface{})[0].(map[string]interface{})
	if result["type"] != "tool_result" || result["tool_use_id"] != "toolu_1" {
		t.Errorf("tool result = %#v, want a tool_result for toolu_1", result)
	}
}

// An error result is flagged rather than thrown, so the model can recover.
func TestClaudeChatMarksErrorResults(t *testing.T) {
	srv, requests := newSequenceServer(t, claudeToolUse, claudeDone)
	chat := newClaudeAt(srv.URL).NewChat("", []Tool{pageTool()})

	if _, err := chat.Send("what does the roadmap say?"); err != nil {
		t.Fatalf("Send returned error: %v", err)
	}
	if _, err := chat.SendToolResults([]ToolResult{{ID: "toolu_1", Content: "no such page", IsError: true}}); err != nil {
		t.Fatalf("SendToolResults returned error: %v", err)
	}

	messages := (*requests)[1]["messages"].([]interface{})
	result := messages[2].(map[string]interface{})["content"].([]interface{})[0].(map[string]interface{})
	if result["is_error"] != true {
		t.Errorf("is_error = %v, want true", result["is_error"])
	}
}

// OpenAI wraps the same definition in a function object.
func TestOpenAIChatSendsToolDefinitions(t *testing.T) {
	srv, requests := newSequenceServer(t, openaiDone)
	chat := newOpenAIAt(srv.URL).NewChat("be brief", []Tool{pageTool()})

	if _, err := chat.Send("what does the roadmap say?"); err != nil {
		t.Fatalf("Send returned error: %v", err)
	}

	tools, ok := (*requests)[0]["tools"].([]interface{})
	if !ok || len(tools) != 1 {
		t.Fatalf("expected one tool on the request, got %#v", (*requests)[0]["tools"])
	}
	tool := tools[0].(map[string]interface{})
	if tool["type"] != "function" {
		t.Errorf("tool type = %v, want function", tool["type"])
	}
	fn := tool["function"].(map[string]interface{})
	if fn["name"] != "get_page" {
		t.Errorf("function name = %v, want get_page", fn["name"])
	}
	if _, ok := fn["parameters"].(map[string]interface{})["properties"]; !ok {
		t.Errorf("schema missing from function.parameters: %#v", fn["parameters"])
	}
}

// OpenAI encodes the arguments as a JSON string; they must come out as the
// same raw object a Claude call gives.
func TestOpenAIChatReportsToolCalls(t *testing.T) {
	srv, _ := newSequenceServer(t, openaiToolUse)
	chat := newOpenAIAt(srv.URL).NewChat("", []Tool{pageTool()})

	resp, err := chat.Send("what does the roadmap say?")
	if err != nil {
		t.Fatalf("Send returned error: %v", err)
	}

	if resp.StopReason != STOPTOOLUSE {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, STOPTOOLUSE)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(resp.ToolCalls))
	}
	args, err := resp.ToolCalls[0].Arguments()
	if err != nil {
		t.Fatalf("Arguments returned error: %v", err)
	}
	if args["page"] != "roadmap" {
		t.Errorf("args = %#v, want page roadmap", args)
	}
}

// The result goes back as a tool message keyed by call id, after the assistant
// turn it answers.
func TestOpenAIChatSendsToolResultWithAskingTurn(t *testing.T) {
	srv, requests := newSequenceServer(t, openaiToolUse, openaiDone)
	chat := newOpenAIAt(srv.URL).NewChat("", []Tool{pageTool()})

	if _, err := chat.Send("what does the roadmap say?"); err != nil {
		t.Fatalf("Send returned error: %v", err)
	}
	if _, err := chat.SendToolResults([]ToolResult{{ID: "call_1", Content: "ship it"}}); err != nil {
		t.Fatalf("SendToolResults returned error: %v", err)
	}

	messages := (*requests)[1]["messages"].([]interface{})
	if len(messages) != 3 {
		t.Fatalf("got %d messages on the second request, want 3", len(messages))
	}

	assistant := messages[1].(map[string]interface{})
	if assistant["role"] != ASSISTANTROLE {
		t.Errorf("second message role = %v, want assistant", assistant["role"])
	}
	if _, ok := assistant["tool_calls"]; !ok {
		t.Errorf("the replayed assistant turn lost its tool_calls: %#v", assistant)
	}

	result := messages[2].(map[string]interface{})
	if result["role"] != TOOLROLE || result["tool_call_id"] != "call_1" {
		t.Errorf("tool message = %#v, want a tool result for call_1", result)
	}
}

// OpenAI has no is_error field, so a failure has to be said in the content.
func TestOpenAIChatMarksErrorResultsInContent(t *testing.T) {
	srv, requests := newSequenceServer(t, openaiToolUse, openaiDone)
	chat := newOpenAIAt(srv.URL).NewChat("", []Tool{pageTool()})

	if _, err := chat.Send("what does the roadmap say?"); err != nil {
		t.Fatalf("Send returned error: %v", err)
	}
	if _, err := chat.SendToolResults([]ToolResult{{ID: "call_1", Content: "no such page", IsError: true}}); err != nil {
		t.Fatalf("SendToolResults returned error: %v", err)
	}

	messages := (*requests)[1]["messages"].([]interface{})
	content := messages[2].(map[string]interface{})["content"]
	if content != "Error: no such page" {
		t.Errorf("content = %v, want the failure spelled out", content)
	}
}

// The loop is the whole point: ask, run, answer, finish.
func TestRunToolLoopRunsToolsUntilDone(t *testing.T) {
	for _, tc := range []struct {
		name      string
		responses []string
		llm       func(url string) LLM
	}{
		{"claude", []string{claudeToolUse, claudeDone}, func(url string) LLM { return newClaudeAt(url) }},
		{"openai", []string{openaiToolUse, openaiDone}, func(url string) LLM { return newOpenAIAt(url) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, requests := newSequenceServer(t, tc.responses...)
			chat := tc.llm(srv.URL).NewChat("be brief", []Tool{pageTool()})

			var ran []string
			resp, err := RunToolLoop(chat, "what does the roadmap say?", func(call ToolCall) ToolResult {
				args, err := call.Arguments()
				if err != nil {
					t.Errorf("Arguments returned error: %v", err)
				}
				ran = append(ran, call.Name+"("+args["page"].(string)+")")
				return ToolResult{ID: call.ID, Content: "ship it"}
			}, 0)
			if err != nil {
				t.Fatalf("RunToolLoop returned error: %v", err)
			}

			if len(ran) != 1 || ran[0] != "get_page(roadmap)" {
				t.Errorf("tools run = %v, want [get_page(roadmap)]", ran)
			}
			if resp.Text != "the roadmap says ship it" {
				t.Errorf("Text = %q", resp.Text)
			}
			if resp.StopReason != STOPEND {
				t.Errorf("StopReason = %q, want %q", resp.StopReason, STOPEND)
			}
			if len(*requests) != 2 {
				t.Errorf("made %d requests, want 2", len(*requests))
			}
		})
	}
}

// A model that never stops asking must not bill forever.
func TestRunToolLoopStopsAtMaxTurns(t *testing.T) {
	srv, requests := newSequenceServer(t, claudeToolUse, claudeToolUse)
	chat := newClaudeAt(srv.URL).NewChat("", []Tool{pageTool()})

	_, err := RunToolLoop(chat, "go", func(call ToolCall) ToolResult {
		return ToolResult{ID: call.ID, Content: "ship it"}
	}, 1)
	if err == nil {
		t.Fatal("expected an error once the turn limit was reached")
	}
	if len(*requests) != 2 {
		t.Errorf("made %d requests, want 2 (one turn past the first reply)", len(*requests))
	}
}

// A tool set is optional: NewChat with none is just a conversation.
func TestChatWithoutToolsSendsNoToolField(t *testing.T) {
	srv, requests := newSequenceServer(t, claudeDone)
	chat := newClaudeAt(srv.URL).NewChat("be brief", nil)

	if _, err := chat.Send("hello"); err != nil {
		t.Fatalf("Send returned error: %v", err)
	}
	if _, ok := (*requests)[0]["tools"]; ok {
		t.Errorf("tools sent on a chat that has none: %#v", (*requests)[0]["tools"])
	}
}
