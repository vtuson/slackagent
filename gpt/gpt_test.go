package gpt

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newOpenAITestServer(t *testing.T, respond func(w http.ResponseWriter)) (*OpenAI, *map[string]interface{}, *http.Header) {
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

	var o OpenAI
	o.SetApiKey("test-key")
	o.SetModel("gpt-3.5-turbo")
	o.SetURL(srv.URL)

	return &o, &body, &header
}

func openAIOK(text string) func(w http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"` + text + `"},"finish_reason":"stop"}]}`))
	}
}

func TestOpenAIGptQueryReturnsText(t *testing.T) {
	o, _, _ := newOpenAITestServer(t, openAIOK("a polite rewrite"))

	got, err := o.GptQuery("be polite", "you are wrong", "")
	if err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if got != "a polite rewrite" {
		t.Errorf("got %q, want %q", got, "a polite rewrite")
	}
}

// OpenAI keeps the system prompt as a message role. This is the structural
// difference from Anthropic, so pin it.
func TestOpenAISystemPromptIsAMessage(t *testing.T) {
	o, body, _ := newOpenAITestServer(t, openAIOK("ok"))

	if _, err := o.GptQuery("be polite", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if _, present := (*body)["system"]; present {
		t.Error("OpenAI requests must not carry a top-level system field")
	}

	messages := (*body)["messages"].([]interface{})
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages without context, got %d", len(messages))
	}
	first := messages[0].(map[string]interface{})
	if first["role"] != SYSTEMROLE || first["content"] != "be polite" {
		t.Errorf("first message = %#v, want a system message with the prompt", first)
	}
}

func TestOpenAIContextAddsThirdMessage(t *testing.T) {
	o, body, _ := newOpenAITestServer(t, openAIOK("ok"))

	if _, err := o.GptQuery("sys", "the message", "the context"); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	messages := (*body)["messages"].([]interface{})
	if len(messages) != 3 {
		t.Fatalf("expected 3 messages with context, got %d", len(messages))
	}
	third := messages[2].(map[string]interface{})
	if third["content"] != "the context" {
		t.Errorf("third message content = %v, want %q", third["content"], "the context")
	}
}

func TestOpenAISendsBearerAuth(t *testing.T) {
	o, _, header := newOpenAITestServer(t, openAIOK("ok"))

	if _, err := o.GptQuery("sys", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if got := header.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want %q", got, "Bearer test-key")
	}
}

func TestOpenAINoChoicesIsAnError(t *testing.T) {
	o, _, _ := newOpenAITestServer(t, func(w http.ResponseWriter) {
		w.Write([]byte(`{"choices":[]}`))
	})

	if _, err := o.GptQuery("sys", "hello", ""); err == nil {
		t.Fatal("expected an error when the response has no choices, got nil")
	}
}

func openAIStop(text, finishReason string) func(w http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"` +
			text + `"},"finish_reason":"` + finishReason + `"}]}`))
	}
}

func TestOpenAIQueryNormalisesStopReasons(t *testing.T) {
	cases := []struct {
		raw  string
		want string
	}{
		{"stop", STOPEND},
		{"length", STOPMAXTOKENS},
		{"tool_calls", STOPTOOLUSE},
		{"function_call", STOPTOOLUSE},
		{"content_filter", STOPREFUSAL},
		{"", STOPOTHER},
	}

	for _, c := range cases {
		o, _, _ := newOpenAITestServer(t, openAIStop("text", c.raw))
		resp, err := o.Query("sys", "hello", "")
		if err != nil {
			t.Fatalf("Query(%s) returned error: %v", c.raw, err)
		}
		if resp.StopReason != c.want {
			t.Errorf("stop reason for %q = %q, want %q", c.raw, resp.StopReason, c.want)
		}
	}
}

// Both providers must report truncation the same way, so a caller's
// continuation loop works regardless of which one is configured.
func TestOpenAIReportsTruncation(t *testing.T) {
	o, _, _ := newOpenAITestServer(t, openAIStop("cut off", "length"))

	resp, err := o.Query("sys", "hello", "")
	if err != nil {
		t.Fatalf("Query returned error: %v", err)
	}
	if !resp.Truncated() {
		t.Error("a length finish_reason must report Truncated() == true")
	}
}

// OpenAI treats max_tokens as optional, so it is only sent when set.
func TestOpenAIMaxTokensOptional(t *testing.T) {
	o, body, _ := newOpenAITestServer(t, openAIOK("ok"))

	if _, err := o.GptQuery("sys", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if _, present := (*body)["max_tokens"]; present {
		t.Error("max_tokens must be omitted when unset, to preserve existing behaviour")
	}

	o.SetMaxTokens(512)
	if _, err := o.GptQuery("sys", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if got := (*body)["max_tokens"]; got != float64(512) {
		t.Errorf("max_tokens = %v, want 512", got)
	}
}

// Neither knob may appear in the request unless it was configured.
func TestOpenAIOmitsUnsetKnobs(t *testing.T) {
	o, body, _ := newOpenAITestServer(t, openAIOK("ok"))

	if _, err := o.GptQuery("", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}

	if _, present := (*body)["temperature"]; present {
		t.Error("temperature was sent without being configured")
	}
	if _, present := (*body)["reasoning_effort"]; present {
		t.Error("reasoning_effort was sent without being configured")
	}
}

func TestOpenAISendsTemperature(t *testing.T) {
	o, body, _ := newOpenAITestServer(t, openAIOK("ok"))
	temp := 1.4 // above Anthropic's ceiling, valid on OpenAI's 0-2 range
	o.SetTemperature(&temp)

	if _, err := o.GptQuery("", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}

	if got := (*body)["temperature"]; got != 1.4 {
		t.Errorf("temperature = %v, want 1.4", got)
	}
}

func TestOpenAISendsZeroTemperature(t *testing.T) {
	o, body, _ := newOpenAITestServer(t, openAIOK("ok"))
	zero := 0.0
	o.SetTemperature(&zero)

	if _, err := o.GptQuery("", "hello", ""); err != nil {
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

// OpenAI's chat completions endpoint takes effort as a top-level
// reasoning_effort, unlike Anthropic's nested output_config.
func TestOpenAISendsReasoningEffort(t *testing.T) {
	o, body, _ := newOpenAITestServer(t, openAIOK("ok"))
	o.SetModel("o3-mini")
	o.SetEffort(EFFORTHIGH)

	if _, err := o.GptQuery("", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}

	if got := (*body)["reasoning_effort"]; got != EFFORTHIGH {
		t.Errorf("reasoning_effort = %v, want %q", got, EFFORTHIGH)
	}
}

// The real shape of the thing the capability table used to pre-empt: the
// model rejects the knob, and the only place that can be explained is on the
// way back out.
func TestOpenAIExplainsRejectedTemperature(t *testing.T) {
	o, _, _ := newOpenAITestServer(t, func(w http.ResponseWriter) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":{"message":"Unsupported parameter: 'temperature' is not supported with this model.","type":"invalid_request_error"}}`))
	})
	o.SetModel("gpt-5")
	temp := 0.7
	o.SetTemperature(&temp)

	_, err := o.Query("sys", "hello", "")
	if err == nil {
		t.Fatal("expected the rejection to surface as an error")
	}
	got := err.Error()
	for _, want := range []string{`"gpt-5"`, `rejected "temperature"`, "gpt config block", "Unsupported parameter"} {
		if !strings.Contains(got, want) {
			t.Errorf("error is missing %q:\n%s", want, got)
		}
	}
}

// Chat turns share post with Query, so this pins that they share the
// explanation too.
func TestOpenAIChatExplainsRejectedEffort(t *testing.T) {
	o, _, _ := newOpenAITestServer(t, func(w http.ResponseWriter) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":{"message":"Unrecognized request argument supplied: reasoning_effort","type":"invalid_request_error"}}`))
	})
	o.SetEffort(EFFORTHIGH)

	_, err := o.NewChat("sys", nil).Send("hello")
	if err == nil {
		t.Fatal("expected the rejection to surface as an error")
	}
	if got := err.Error(); !strings.Contains(got, `rejected "effort"`) {
		t.Errorf("a chat turn must explain the knob too:\n%s", got)
	}
}

// An ordinary failure must not be dressed up as a knob problem.
func TestOpenAILeavesUnrelatedErrorsAlone(t *testing.T) {
	o, _, _ := newOpenAITestServer(t, func(w http.ResponseWriter) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}`))
	})
	temp := 0.7
	o.SetTemperature(&temp)

	_, err := o.Query("sys", "hello", "")
	if err == nil {
		t.Fatal("expected an error")
	}
	if strings.Contains(err.Error(), "gpt config block") {
		t.Errorf("a bad key must not be blamed on temperature:\n%s", err)
	}
}
