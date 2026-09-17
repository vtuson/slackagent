package gpt

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newClaudeTestServer returns a Claude pointed at a stub server, plus a pointer
// to the decoded request body the stub last received.
func newClaudeTestServer(t *testing.T, respond func(w http.ResponseWriter)) (*Claude, *map[string]interface{}, *http.Header) {
	t.Helper()

	var body map[string]interface{}
	var header http.Header

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header = r.Header.Clone()
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading request body: %v", err)
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("unmarshalling request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		respond(w)
	}))
	t.Cleanup(srv.Close)

	var c Claude
	c.SetApiKey("test-key")
	c.SetModel("claude-opus-5")
	c.SetURL(srv.URL)

	return &c, &body, &header
}

func okResponse(text string) func(w http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		w.Write([]byte(`{
			"id": "msg_test",
			"type": "message",
			"role": "assistant",
			"model": "claude-opus-5",
			"content": [{"type": "text", "text": "` + text + `"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`))
	}
}

func TestClaudeGptQueryReturnsText(t *testing.T) {
	c, _, _ := newClaudeTestServer(t, okResponse("a polite rewrite"))

	got, err := c.GptQuery("be polite", "you are wrong", "")
	if err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if got != "a polite rewrite" {
		t.Errorf("got %q, want %q", got, "a polite rewrite")
	}
}

// The system prompt must go in Anthropic's top-level "system" field, not as a
// message with a system role like OpenAI uses.
func TestClaudeSystemPromptIsTopLevel(t *testing.T) {
	c, body, _ := newClaudeTestServer(t, okResponse("ok"))

	if _, err := c.GptQuery("be polite", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}

	system, ok := (*body)["system"].([]interface{})
	if !ok || len(system) != 1 {
		t.Fatalf("expected a single top-level system block, got %#v", (*body)["system"])
	}
	if text := system[0].(map[string]interface{})["text"]; text != "be polite" {
		t.Errorf("system text = %v, want %q", text, "be polite")
	}

	messages := (*body)["messages"].([]interface{})
	for _, m := range messages {
		if role := m.(map[string]interface{})["role"]; role == "system" {
			t.Error("system prompt leaked into messages array; Anthropic rejects a system role there")
		}
	}
}

// Context is folded into the single user turn rather than sent as a second
// user message, so we never depend on turn-merging behaviour.
func TestClaudeContextFoldedIntoUserTurn(t *testing.T) {
	c, body, _ := newClaudeTestServer(t, okResponse("ok"))

	if _, err := c.GptQuery("sys", "the message", "the context"); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}

	messages := (*body)["messages"].([]interface{})
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	content := messages[0].(map[string]interface{})["content"].([]interface{})
	text := content[0].(map[string]interface{})["text"].(string)
	if !strings.Contains(text, "the message") || !strings.Contains(text, "the context") {
		t.Errorf("user turn %q should contain both message and context", text)
	}
}

// max_tokens is required by the Messages API; OpenAI treats it as optional.
func TestClaudeSendsMaxTokens(t *testing.T) {
	c, body, _ := newClaudeTestServer(t, okResponse("ok"))

	if _, err := c.GptQuery("sys", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if got := (*body)["max_tokens"]; got != float64(CLAUDEMAXTOKENS) {
		t.Errorf("max_tokens = %v, want %d", got, CLAUDEMAXTOKENS)
	}

	c.SetMaxTokens(64)
	if _, err := c.GptQuery("sys", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if got := (*body)["max_tokens"]; got != float64(64) {
		t.Errorf("max_tokens = %v, want 64", got)
	}
}

func TestClaudeSendsAuthHeaders(t *testing.T) {
	c, _, header := newClaudeTestServer(t, okResponse("ok"))

	if _, err := c.GptQuery("sys", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if got := header.Get("x-api-key"); got != "test-key" {
		t.Errorf("x-api-key = %q, want %q", got, "test-key")
	}
	if got := header.Get("anthropic-version"); got == "" {
		t.Error("anthropic-version header is required but was not sent")
	}
}

func TestClaudeRefusalIsAnError(t *testing.T) {
	c, _, _ := newClaudeTestServer(t, func(w http.ResponseWriter) {
		w.Write([]byte(`{
			"id": "msg_test",
			"type": "message",
			"role": "assistant",
			"model": "claude-opus-5",
			"content": [],
			"stop_reason": "refusal",
			"stop_details": {"type": "refusal", "category": "cyber", "explanation": "declined"},
			"usage": {"input_tokens": 10, "output_tokens": 0}
		}`))
	})

	if _, err := c.GptQuery("sys", "hello", ""); err == nil {
		t.Fatal("expected an error for a refusal stop_reason, got nil")
	}
}

func TestClaudeEmptyContentIsAnError(t *testing.T) {
	c, _, _ := newClaudeTestServer(t, func(w http.ResponseWriter) {
		w.Write([]byte(`{
			"id": "msg_test",
			"type": "message",
			"role": "assistant",
			"model": "claude-opus-5",
			"content": [],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 0}
		}`))
	})

	if _, err := c.GptQuery("sys", "hello", ""); err == nil {
		t.Fatal("expected an error for an empty content array, got nil")
	}
}

func TestClaudeUsesConfiguredModel(t *testing.T) {
	c, body, _ := newClaudeTestServer(t, okResponse("ok"))
	c.SetModel("claude-haiku-4-5")

	if _, err := c.GptQuery("sys", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if got := (*body)["model"]; got != "claude-haiku-4-5" {
		t.Errorf("model = %v, want claude-haiku-4-5", got)
	}
}
