package agent

import (
	"testing"

	"github.com/vtuson/slackagent/gpt"
)

// newTestAgent builds an Agent with just the GPT config populated, bypassing
// LoadConfig (which exits the process when Slack tokens are absent).
func newTestAgent(key, model, provider string) *Agent {
	return newTestAgentMaxTokens(key, model, provider, 0)
}

func newTestAgentMaxTokens(key, model, provider string, maxTokens int64) *Agent {
	cfg := &Config{}
	cfg.GPT = &GPTConfig{Key: key, Model: model, Provider: provider, MaxTokens: maxTokens}

	return &Agent{Config: cfg}
}

func newTestAgentKnobs(key, model, provider string, temperature *float64, effort string) *Agent {
	cfg := &Config{}
	cfg.GPT = &GPTConfig{
		Key:         key,
		Model:       model,
		Provider:    provider,
		Temperature: temperature,
		Effort:      effort,
	}

	return &Agent{Config: cfg}
}

func TestNewLLMExplicitProvider(t *testing.T) {
	a := newTestAgent("k", "claude-opus-5", gpt.PROVIDERANTHROPIC)
	if _, ok := a.NewLLM().(*gpt.Claude); !ok {
		t.Errorf("expected *gpt.Claude for an anthropic provider, got %T", a.NewLLM())
	}

	a = newTestAgent("k", "gpt-3.5-turbo", gpt.PROVIDEROPENAI)
	if _, ok := a.NewLLM().(*gpt.OpenAI); !ok {
		t.Errorf("expected *gpt.OpenAI for an openai provider, got %T", a.NewLLM())
	}
}

// Configs written before the provider field existed must keep working.
func TestNewLLMBackwardsCompatible(t *testing.T) {
	a := newTestAgent("k", "gpt-3.5-turbo", "")
	if _, ok := a.NewLLM().(*gpt.OpenAI); !ok {
		t.Errorf("a config with no provider and a gpt model must stay on OpenAI, got %T", a.NewLLM())
	}
}

// A claude model with no provider set should be inferred.
func TestNewLLMInfersAnthropic(t *testing.T) {
	a := newTestAgent("k", "claude-haiku-4-5", "")
	if _, ok := a.NewLLM().(*gpt.Claude); !ok {
		t.Errorf("a claude model should infer the anthropic provider, got %T", a.NewLLM())
	}
}

// An empty model must default per provider, not to the OpenAI default.
func TestNewLLMDefaultsModelPerProvider(t *testing.T) {
	a := newTestAgent("k", "", gpt.PROVIDERANTHROPIC)
	a.NewLLM()
	if a.Config.GPT.Model != gpt.MODELCLAUDE {
		t.Errorf("model = %q, want %q", a.Config.GPT.Model, gpt.MODELCLAUDE)
	}

	a = newTestAgent("k", "", gpt.PROVIDEROPENAI)
	a.NewLLM()
	if a.Config.GPT.Model != gpt.GetDefaultModel() {
		t.Errorf("model = %q, want %q", a.Config.GPT.Model, gpt.GetDefaultModel())
	}
}

func TestNewEmbedderRejectsAnthropic(t *testing.T) {
	a := newTestAgent("k", "claude-opus-5", gpt.PROVIDERANTHROPIC)
	if _, err := a.NewEmbedder(); err == nil {
		t.Fatal("expected an error: anthropic has no embeddings endpoint")
	}

	a = newTestAgent("k", "gpt-3.5-turbo", gpt.PROVIDEROPENAI)
	if _, err := a.NewEmbedder(); err != nil {
		t.Fatalf("openai embedder should work: %v", err)
	}
}

// max_tokens from the config file must reach the provider, and an unset value
// must leave the provider on its own default.
func TestNewLLMAppliesMaxTokens(t *testing.T) {
	a := newTestAgentMaxTokens("k", "claude-opus-5", gpt.PROVIDERANTHROPIC, 4096)
	if got := a.NewLLM().(*gpt.Claude).MaxTokens(); got != 4096 {
		t.Errorf("claude max tokens = %d, want 4096", got)
	}

	a = newTestAgentMaxTokens("k", "claude-opus-5", gpt.PROVIDERANTHROPIC, 0)
	if got := a.NewLLM().(*gpt.Claude).MaxTokens(); got != 0 {
		t.Errorf("an unset max_tokens must stay 0 so the provider default applies, got %d", got)
	}

	a = newTestAgentMaxTokens("k", "gpt-3.5-turbo", gpt.PROVIDEROPENAI, 512)
	if got := a.NewLLM().(*gpt.OpenAI).MaxTokens(); got != 512 {
		t.Errorf("openai max tokens = %d, want 512", got)
	}
}

// temperature and effort from the config file must reach the provider.
// The unsupported combinations are rejected by gpt.ApplyKnobs, which NewLLM
// treats as fatal, so only the supported pairings are exercised here.
func TestNewLLMAppliesKnobs(t *testing.T) {
	a := newTestAgentKnobs("k", "claude-opus-5", gpt.PROVIDERANTHROPIC, nil, gpt.EFFORTMEDIUM)
	if got := a.NewLLM().(*gpt.Claude).Effort(); got != gpt.EFFORTMEDIUM {
		t.Errorf("claude effort = %q, want %q", got, gpt.EFFORTMEDIUM)
	}

	temp := 0.3
	a = newTestAgentKnobs("k", "gpt-4o", gpt.PROVIDEROPENAI, &temp, "")
	got := a.NewLLM().(*gpt.OpenAI).Temperature()
	if got == nil || *got != temp {
		t.Errorf("openai temperature = %v, want %v", got, temp)
	}
}

// An absent temperature must stay absent rather than becoming zero, which
// would quietly pin the model to its most deterministic setting.
func TestNewLLMLeavesUnsetKnobsAlone(t *testing.T) {
	a := newTestAgentKnobs("k", "gpt-4o", gpt.PROVIDEROPENAI, nil, "")
	llm := a.NewLLM().(*gpt.OpenAI)
	if llm.Temperature() != nil {
		t.Errorf("temperature = %v, want nil", llm.Temperature())
	}
	if llm.Effort() != "" {
		t.Errorf("effort = %q, want empty", llm.Effort())
	}
}
