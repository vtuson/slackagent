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

// stopReasonResponse returns a stub reply carrying a given stop reason.
func stopReasonResponse(text, stopReason string) func(w http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		w.Write([]byte(`{
			"id": "msg_test",
			"type": "message",
			"role": "assistant",
			"model": "claude-opus-5",
			"content": [{"type": "text", "text": "` + text + `"}],
			"stop_reason": "` + stopReason + `",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`))
	}
}

func TestClaudeQueryNormalisesStopReasons(t *testing.T) {
	cases := []struct {
		raw  string
		want string
	}{
		{"end_turn", STOPEND},
		{"stop_sequence", STOPEND},
		{"max_tokens", STOPMAXTOKENS},
		{"tool_use", STOPTOOLUSE},
		{"refusal", STOPREFUSAL},
		{"pause_turn", STOPOTHER},
	}

	for _, c := range cases {
		claude, _, _ := newClaudeTestServer(t, stopReasonResponse("partial", c.raw))
		resp, err := claude.Query("sys", "hello", "")
		if err != nil {
			t.Fatalf("Query(%s) returned error: %v", c.raw, err)
		}
		if resp.StopReason != c.want {
			t.Errorf("stop reason for %q = %q, want %q", c.raw, resp.StopReason, c.want)
		}
		if resp.RawStopReason != c.raw {
			t.Errorf("raw stop reason = %q, want %q", resp.RawStopReason, c.raw)
		}
	}
}

// Truncated is the signal a caller uses to ask for a continuation.
func TestClaudeQueryReportsTruncation(t *testing.T) {
	claude, _, _ := newClaudeTestServer(t, stopReasonResponse("cut off mid-", "max_tokens"))

	resp, err := claude.Query("sys", "write an essay", "")
	if err != nil {
		t.Fatalf("Query returned error: %v", err)
	}
	if !resp.Truncated() {
		t.Error("a max_tokens stop reason must report Truncated() == true")
	}
	if resp.Text != "cut off mid-" {
		t.Errorf("truncated text should still be returned, got %q", resp.Text)
	}

	claude, _, _ = newClaudeTestServer(t, stopReasonResponse("all done", "end_turn"))
	resp, _ = claude.Query("sys", "hi", "")
	if resp.Truncated() {
		t.Error("an end_turn stop reason must report Truncated() == false")
	}
}

// A refusal is data on Query and an error on GptQuery.
func TestClaudeQueryReturnsRefusalAsData(t *testing.T) {
	refusal := func(w http.ResponseWriter) {
		w.Write([]byte(`{
			"id": "msg_test", "type": "message", "role": "assistant",
			"model": "claude-opus-5", "content": [], "stop_reason": "refusal",
			"stop_details": {"type": "refusal", "category": "cyber", "explanation": "declined"},
			"usage": {"input_tokens": 10, "output_tokens": 0}
		}`))
	}

	claude, _, _ := newClaudeTestServer(t, refusal)
	resp, err := claude.Query("sys", "hello", "")
	if err != nil {
		t.Fatalf("Query should not error on a refusal: %v", err)
	}
	if resp.StopReason != STOPREFUSAL {
		t.Errorf("stop reason = %q, want %q", resp.StopReason, STOPREFUSAL)
	}
	if resp.Detail != "declined" {
		t.Errorf("Detail = %q, want the refusal explanation", resp.Detail)
	}

	claude, _, _ = newClaudeTestServer(t, refusal)
	if _, err := claude.GptQuery("sys", "hello", ""); err == nil {
		t.Error("GptQuery must still surface a refusal as an error")
	}
}

// The use case the stop reason exists for: continue until the model stops
// because it is finished rather than because it ran out of room.
func TestClaudeContinuationLoop(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		calls++
		if calls < 3 {
			stopReasonResponse("chunk ", "max_tokens")(w)
			return
		}
		stopReasonResponse("end", "end_turn")(w)
	}))
	defer srv.Close()

	var c Claude
	c.SetApiKey("k")
	c.SetModel("claude-opus-5")
	c.SetURL(srv.URL)

	var full string
	for i := 0; i < 10; i++ {
		resp, err := c.Query("sys", "write a long thing", full)
		if err != nil {
			t.Fatalf("Query returned error: %v", err)
		}
		full += resp.Text
		if !resp.Truncated() {
			break
		}
	}

	if calls != 3 {
		t.Errorf("expected the loop to stop after 3 calls, made %d", calls)
	}
	if full != "chunk chunk end" {
		t.Errorf("assembled text = %q, want %q", full, "chunk chunk end")
	}
}

func TestClaudeMaxTokensOverride(t *testing.T) {
	c, body, _ := newClaudeTestServer(t, okResponse("ok"))
	c.SetMaxTokens(4096)

	if _, err := c.GptQuery("sys", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if got := (*body)["max_tokens"]; got != float64(4096) {
		t.Errorf("max_tokens = %v, want 4096", got)
	}
}

// Neither knob may appear in the request unless it was configured, otherwise
// every caller would be silently opted into a default they never chose.
func TestClaudeOmitsUnsetKnobs(t *testing.T) {
	c, body, _ := newClaudeTestServer(t, okResponse("ok"))

	if _, err := c.GptQuery("", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}

	if _, present := (*body)["temperature"]; present {
		t.Error("temperature was sent without being configured")
	}
	if _, present := (*body)["output_config"]; present {
		t.Error("output_config was sent without an effort being configured")
	}
}

// Effort goes inside output_config, not at the top level.
func TestClaudeSendsEffortInOutputConfig(t *testing.T) {
	c, body, _ := newClaudeTestServer(t, okResponse("ok"))
	c.SetEffort(EFFORTMEDIUM)

	if _, err := c.GptQuery("", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}

	if _, present := (*body)["effort"]; present {
		t.Error("effort was sent top-level; Anthropic expects it inside output_config")
	}
	outputConfig, ok := (*body)["output_config"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected an output_config object, got %#v", (*body)["output_config"])
	}
	if got := outputConfig["effort"]; got != EFFORTMEDIUM {
		t.Errorf("output_config.effort = %v, want %q", got, EFFORTMEDIUM)
	}
}

func TestClaudeSendsTemperature(t *testing.T) {
	c, body, _ := newClaudeTestServer(t, okResponse("ok"))
	c.SetModel("claude-opus-4-6") // an older model that still accepts sampling
	temp := 0.4
	c.SetTemperature(&temp)

	if _, err := c.GptQuery("", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}

	if got := (*body)["temperature"]; got != 0.4 {
		t.Errorf("temperature = %v, want 0.4", got)
	}
}

// A zero temperature must reach the wire. The SDK omits zero-valued fields, so
// this pins that the param.Opt wrapper is doing its job.
func TestClaudeSendsZeroTemperature(t *testing.T) {
	c, body, _ := newClaudeTestServer(t, okResponse("ok"))
	c.SetModel("claude-opus-4-6")
	zero := 0.0
	c.SetTemperature(&zero)

	if _, err := c.GptQuery("", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}

	got, present := (*body)["temperature"]
	if !present {
		t.Fatal("temperature 0 never reached the request body")
	}
	if got != 0.0 {
		t.Errorf("temperature = %v, want 0", got)
	}
}
