"""The prompt box's reader: rules, the model's answer laid over them, the fact checks, and the ways the model
can fail. The model is a fake; nothing here reaches a network."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from src.agents.video import brief as brief_agent
from src.agents.video import fact_ledger, policy, prompt_brief
from src.core.exceptions import IngestionError, ModelOutputParseError, ModelProviderError
from src.data.ingestion import load_property_input
from src.data.schemas import PropertyKnowledge
from src.data.video_schemas import PromptBrief, VideoInput
from src.models import gateway
from tests.video.helpers import make_video_property


@pytest.fixture(autouse=True)
def model(monkeypatch):
    """Stands in for the gateway. By default it answers like a machine with no key; a test sets `answer` or
    `raises`. Every call is recorded."""
    state = SimpleNamespace(calls=[], answer=None, raises=ModelProviderError("GEMINI_API_KEY is not set."), delay=0.0,
                            policies=[])

    def fake(role, fn_name, prompt, *args, **kwargs):
        state.calls.append((role, fn_name, prompt))
        state.policies.append(gateway._call_policy.get())
        if state.delay:
            time.sleep(state.delay)
        if state.raises:
            raise state.raises
        return state.answer

    monkeypatch.setattr(prompt_brief, "call_with_fallback", fake)
    return state


@pytest.fixture
def folder(tmp_path):
    return make_video_property(tmp_path / "data" / "properties")


def read(prompt, folder, **kwargs):
    return prompt_brief.read_prompt(prompt, folder, offline=True, **kwargs)


def answers(model, **fields):
    model.raises, model.answer = None, fields


# -- length ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("prompt,seconds", [
    ("A 10 second video", 10), ("make it 15-second", 15), ("a 30s reel", 30), ("45 sec please", 45),
    ("a thirty second reel", 30), ("forty-five seconds", 45), ("sixty seconds", 60), ("a minute long", 60),
    ("one minute", 60), ("half a minute", 30), ("a 1 minute film", 60), ("a teaser", 10)])
def test_a_length_the_prompt_names_is_read_in_seconds(prompt, seconds, folder):
    result = read(prompt, folder)
    assert result.brief.duration_s == seconds
    assert not [n for n in result.notes if "not offered" in n]


@pytest.mark.parametrize("prompt,seconds,said", [
    ("a 20 second video", 15, "Length 20 s is not offered; using 15 s"),
    ("a 50 sec video", 45, "Length 50 s is not offered; using 45 s"),
    ("a 90 second video", 60, "Length 90 s is not offered; using 60 s"),
    ("a 5 second video", 10, "Length 5 s is not offered; using 10 s"),
    ("two minutes long", 60, "Length 120 s is not offered; using 60 s")])
def test_a_length_that_is_not_offered_snaps_to_the_nearest_with_a_note(prompt, seconds, said, folder):
    result = read(prompt, folder)
    assert result.brief.duration_s == seconds
    assert any(said in n for n in result.notes)


def test_a_distance_or_a_floor_is_not_a_length(folder):
    result = read("Two minutes from the metro, on the thirty second floor, 5 minutes by car", folder)
    assert result.brief.duration_s is None


def test_several_lengths_use_the_first_and_say_so(folder):
    result = read("A 10 second teaser and a 30 second version", folder)
    assert result.brief.duration_s == 10
    assert any("Several lengths are named (10 s, 30 s); using the first." in n for n in result.notes)


# -- shape ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("prompt,aspect", [
    ("a vertical video", "9:16"), ("an Instagram reel", "9:16"), ("for YouTube Shorts", "9:16"), ("9:16 please", "9:16"),
    ("a landscape film", "16:9"), ("for YouTube", "16:9"), ("widescreen", "16:9"), ("16:9", "16:9"),
    ("a square video", "1:1"), ("1:1 for the feed", "1:1")])
def test_a_shape_the_prompt_names_is_read(prompt, aspect, folder):
    assert read(prompt, folder).brief.aspect == aspect


def test_two_shapes_use_the_first_and_say_so(folder):
    result = read("A vertical reel and a square one", folder)
    assert result.brief.aspect == "9:16"
    assert any("Several shapes are named (9:16, 1:1); using 9:16, the first." in n for n in result.notes)


def test_a_prompt_that_names_neither_leaves_length_and_shape_unset(folder):
    result = read("A calm video for families", folder)
    assert result.brief.duration_s is None and result.brief.aspect is None


# -- the other rule families ------------------------------------------------------------------------------

@pytest.mark.parametrize("prompt,objective", [
    ("book a site visit", "site_visit"), ("a walkthrough of the model flat", "site_visit"),
    ("now launching in Baner", "launch"), ("introducing the new tower", "launch"),
    ("drive enquiries and leads", "enquiry"), ("get in touch with the sales team", "enquiry"),
    ("a brand film for awareness", "awareness")])
def test_the_objective_comes_from_its_keywords(prompt, objective, folder):
    assert read(prompt, folder).brief.objective == objective


def test_the_objective_with_most_mentions_wins_and_none_means_unset(folder):
    assert read("launch launch launch, and maybe a visit", folder).brief.objective == "launch"
    assert read("a nice video", folder).brief.objective is None


@pytest.mark.parametrize("prompt,tone", [
    ("a warm video", "warm"), ("friendly and welcoming", "warm"), ("a confident, bold film", "confident"),
    ("an elegant, luxurious feel", "refined"), ("refined", "refined")])
def test_the_tone_comes_from_its_name_or_a_synonym(prompt, tone, folder):
    assert read(prompt, folder).brief.tone == tone


@pytest.mark.parametrize("prompt,audience", [
    ("for young families", "young families"), ("aimed at first-time buyers", "first-time buyers"),
    ("for investors", "investors"), ("for NRIs", "NRIs"), ("for families", "families")])
def test_the_audience_is_a_known_group(prompt, audience, folder):
    assert read(prompt, folder).brief.audience == audience


@pytest.mark.parametrize("prompt,mood", [
    ("with calm music", "calm"), ("warm acoustic music", "warm acoustic"), ("an upbeat track", "upbeat"),
    ("soft piano background music", "soft piano"), ("cinematic score", "cinematic")])
def test_the_music_mood_is_the_words_before_music(prompt, mood, folder):
    assert read(prompt, folder).brief.music_mood == mood


def test_captions_off_and_on(folder):
    assert read("no captions please", folder).brief.burn_captions is False
    assert read("without subtitles", folder).brief.burn_captions is False
    assert read("add captions", folder).brief.burn_captions is True
    assert read("a video", folder).brief.burn_captions is None


@pytest.mark.parametrize("prompt,cta", [
    ('End with "Book a site visit this weekend".', "Book a site visit this weekend"),
    ('CTA: "Visit us today"', "Visit us today"), ("CTA: Book a visit today.", "Book a visit today"),
    ("call to action: Enquire now", "Enquire now"), ('Closing with “Come and see it”', "Come and see it")])
def test_a_call_to_action_is_only_what_the_prompt_words_after_a_cue(prompt, cta, folder):
    assert read(prompt, folder).brief.cta == cta


def test_a_quoted_line_with_no_cue_is_not_a_call_to_action(folder):
    result = read('Overlay "Where Nature Meets Luxury" on the first shot', folder)
    assert result.brief.cta is None
    assert any("Quoted lines are not printed (1)" in n for n in result.notes)


def test_the_rest_of_the_prompt_is_creative_direction(folder):
    result = read("A calm 30 second reel. Soft evening light and an unhurried pace, no rush at all.", folder)
    assert result.brief.creative_direction == "Calm. Soft evening light and an unhurried pace, no rush at all."


def test_what_the_rules_have_read_is_not_repeated_as_direction(folder):
    prompt = ('Make a calm 30 second vertical reel for young families. Highlight the swimming pool, gym and clubhouse, '
              'with warm acoustic music. End with "Book a site visit this weekend". Keep it elegant and unhurried, '
              'golden evening light.')
    result = read(prompt, folder)
    assert result.brief.creative_direction == "Calm. Keep it elegant and unhurried, golden evening light."
    assert (result.brief.duration_s, result.brief.aspect, result.brief.audience) == (30, "9:16", "young families")


def test_a_prompt_the_rules_have_read_entirely_leaves_no_direction(folder):
    for prompt in ("Make a 30 second vertical video for young families", "A 45 sec teaser to launch it",
                   "A 10 second reel for investors, no captions, with calm music"):
        assert read(prompt, folder).brief.creative_direction is None, prompt


def test_a_clause_that_is_only_read_words_is_dropped_and_the_rest_kept(folder):
    result = read("A square video for investors, with a dreamy look at dusk", folder)
    assert result.brief.creative_direction == "Dreamy look at dusk."


def test_what_the_video_cannot_do_is_named_and_kept_out_of_the_direction(folder):
    result = read("Cinematic drone shot over the towers in 8k. Keep the mood relaxed and slow throughout.", folder)
    assert result.brief.creative_direction == "Keep the mood relaxed and slow throughout."
    assert any("Camera moves in the prompt are not used" in n and "drone" in n for n in result.notes)
    assert any("Resolution and render-quality words are not used" in n and "8k" in n for n in result.notes)


# -- the highlights and the facts --------------------------------------------------------------------------

def test_a_highlight_is_the_properties_own_wording(folder):
    result = read("Highlight the lap pool, the gym and the clubhouse", folder)
    assert result.brief.highlights == ["Swimming pool", "Gymnasium", "Clubhouse"]
    assert not [n for n in result.notes if "left out" in n]


def test_an_amenity_the_property_does_not_have_is_left_out_with_a_plain_note(folder):
    result = read("Highlight the rooftop helipad, the pool and the gym", folder)
    assert result.brief.highlights == ["Swimming pool", "Gymnasium"]
    assert "'rooftop helipad' is not in this property's details; left out." in result.notes


def test_the_tag_vocabulary_matches_wording_but_never_admits_a_highlight(tmp_path):
    no_pool = make_video_property(tmp_path / "data" / "properties", amenities="Clubhouse\nGymnasium")
    result = read("Highlight the swimming pool, the jacuzzi and the gym", no_pool)
    assert result.brief.highlights == ["Gymnasium"]
    assert "'swimming pool' is not in this property's details; left out." in result.notes
    assert "'jacuzzi' is not in this property's details; left out." in result.notes


def test_a_highlight_that_joins_two_things_the_property_has_separately_is_left_out(tmp_path):
    folder = make_video_property(tmp_path / "data" / "properties", amenities="Swimming pool\nRooftop lounge")
    result = read("Highlight the rooftop pool", folder)
    assert result.brief.highlights is None
    assert "'rooftop pool' is not in this property's details; left out." in result.notes


def test_a_selling_point_can_be_a_highlight(folder):
    result = read("Feature the metro station", folder)
    assert result.brief.highlights == ["Two minutes from the Baner metro station"]


def test_a_highlight_with_a_figure_that_is_not_a_fact_is_left_out(folder, model):
    answers(model, highlights=["Clubhouse of 40,000 sq ft", "Clubhouse of 25,000 sq ft", "Swimming pool"])
    result = prompt_brief.read_prompt("highlight the clubhouse and pool", folder)
    assert result.brief.highlights == ["Clubhouse", "Swimming pool"]       # 25,000 is a fact, 40,000 is not
    assert "'Clubhouse of 40,000 sq ft' has a figure (40,000) that is not in this property's facts; left out." \
        in result.notes


def test_a_call_to_action_with_a_figure_that_is_not_a_fact_is_left_out(folder):
    result = read('CTA: "Call 99999 00000 now"', folder)
    assert result.brief.cta is None
    assert "The call to action has a figure (00000) that is not in this property's facts; left out." in result.notes


def test_a_number_word_is_a_figure_too(folder):
    result = read('End with "Get twenty percent off"', folder)
    assert result.brief.cta is None and any("figure (twenty)" in n for n in result.notes)


def test_a_figure_the_property_has_is_allowed(folder):
    assert read('CTA: "Call 98765 43210"', folder).brief.cta == "Call 98765 43210"


def test_a_sentence_with_a_price_is_not_passed_on_as_direction(folder):
    result = read("A calm reel for families. Priced at Rs 49 lakh, 20% off this week.", folder)
    assert result.brief.creative_direction == "Calm."
    assert any("Not used as creative direction" in n and "Rs 49 lakh" in n for n in result.notes)


def test_the_cached_ledger_adds_what_a_document_states(folder, tmp_path):
    prop = load_property_input(folder)
    knowledge = PropertyKnowledge(facts={"amenities": "Swimming pool, Helipad"}, selling_points=[])
    path = fact_ledger.cache_dir() / f"{fact_ledger.cache_key(prop, '', fact_ledger.model_name(), fact_ledger.prompt_version())}.json"
    fact_ledger._store(path, knowledge, fact_ledger.model_name(), fact_ledger.prompt_version())

    assert read("Highlight the helipad", folder).brief.highlights == ["Helipad"]
    assert read("Highlight the helipad", folder).brief.highlights == ["Helipad"]      # and nothing is asked of a model
    path.unlink()
    assert read("Highlight the helipad", folder).brief.highlights is None


def test_things_a_prompt_only_describes_are_not_highlights(folder):
    result = read("Lush green trees and landscaped gardens, for families with kids", folder)
    assert result.brief.highlights is None


def test_the_amenities_a_prompt_names_without_a_cue_are_taken_when_the_property_has_them(folder):
    result = read("A reel of the swimming pool, the gym and a helipad", folder)
    assert result.brief.highlights == ["Swimming pool", "Gymnasium"]
    assert not [n for n in result.notes if "left out" in n]               # no cue: no claim to refuse


def test_a_highlight_list_ends_where_a_clause_begins(folder):
    result = read("Highlight the pool and gym for young families, with calm music", folder)
    assert result.brief.highlights == ["Swimming pool", "Gymnasium"]
    assert result.brief.music_mood == "calm"
    assert not [n for n in result.notes if "left out" in n]


def test_a_generic_item_is_not_a_highlight_and_not_refused(folder):
    result = read("Highlight the property and its amenities", folder)
    assert result.brief.highlights is None and not [n for n in result.notes if "left out" in n]


# -- the model ----------------------------------------------------------------------------------------------

def test_the_model_asked_with_the_role_and_nothing_but_text(folder, model):
    answers(model)
    prompt_brief.read_prompt("A calm reel for families", folder)

    (role, fn_name, text), = model.calls
    assert (role, fn_name) == ("creative_director", "generate_json")
    assert "Aurelia Heights (mid)" in text and "- Swimming pool" in text and "- Two minutes from the Baner metro station" in text
    assert "site_visit" in text and "refined" in text and "A calm reel for families" in text
    assert not [p for p in ("$name", "$category", "$things", "$objectives", "$tones", "$prompt") if p in text]


def test_the_model_overrides_a_rule_only_where_it_returns_a_value(folder, model):
    answers(model, objective="launch", tone="refined", audience=None, highlights=["Gymnasium"], cta=None,
            creative_direction="Unhurried, evening light.", music_mood=None, burn_captions=False)
    result = prompt_brief.read_prompt("Book a site visit for young families, warm. Highlight the pool.", folder)

    assert result.source == "model"
    brief = result.brief
    assert (brief.objective, brief.tone) == ("launch", "refined")          # the model's, over the rules' site_visit, warm
    assert brief.audience == "young families"                              # the model said null: the rule's stays
    assert brief.highlights == ["Gymnasium"] and brief.creative_direction == "Unhurried, evening light."
    assert brief.burn_captions is False
    assert not [n for n in result.notes if "model was not used" in n]


def test_an_empty_answer_changes_nothing_and_is_still_the_models(folder, model):
    answers(model, objective=None, highlights=[], pronunciations={})
    result = prompt_brief.read_prompt("A video to launch the project", folder)
    assert result.source == "model" and result.brief.objective == "launch"
    assert result.brief.highlights is None and result.brief.pronunciations is None


def test_a_pronunciation_comes_from_the_model_only(folder, model):
    answers(model, pronunciations={"Baner": "Bah-ner", " ": "x"})
    assert prompt_brief.read_prompt("say Baner as Bah-ner", folder).brief.pronunciations == {"Baner": "Bah-ner"}


def test_a_value_that_is_not_one_of_the_choices_is_dropped_with_a_note(folder, model):
    answers(model, objective="brand love", tone="loud", audience=["a list"], burn_captions="yes", highlights="Pool")
    result = prompt_brief.read_prompt("A video to launch it, warm", folder)

    assert (result.brief.objective, result.brief.tone) == ("launch", "warm")      # the rules' values stand
    assert result.brief.audience is None and result.brief.burn_captions is None
    for said in ("The model's objective ('brand love') is not one of", "The model's tone ('loud') is not one of",
                 "The model's audience was not usable; left out.", "The model's captions choice was not usable",
                 "The model's highlights was not usable; left out."):
        assert any(said in n for n in result.notes), said


@pytest.mark.parametrize("answer", [["not", "an", "object"], "text", 5])
def test_an_answer_that_is_not_an_object_falls_back_with_a_note(answer, folder, model):
    model.raises, model.answer = None, answer
    result = prompt_brief.read_prompt("A video to launch it", folder)
    assert result.brief.objective == "launch"
    assert "The model's answer was not a JSON object; the rules were used." in result.notes


def test_bad_json_falls_back_to_the_rules_with_a_note(folder, model):
    model.raises = ModelOutputParseError("Could not parse JSON from model output: 'sorry' (an API key is not set)")
    result = prompt_brief.read_prompt("A video to launch it", folder)
    assert result.source == "rules" and result.brief.objective == "launch"
    assert "The model was not used (its answer was not JSON); the prompt was read with built-in rules." in result.notes


def test_offline_never_calls_the_model(folder, model):
    answers(model, objective="enquiry")
    result = prompt_brief.read_prompt("A video to launch it", folder, offline=True)
    assert model.calls == [] and result.source == "rules" and result.brief.objective == "launch"
    assert "The model was not used (an offline run); the prompt was read with built-in rules." in result.notes


def test_an_empty_prompt_asks_nothing(folder, model):
    result = prompt_brief.read_prompt("   ", folder)
    assert model.calls == [] and result.notes == ["The prompt is empty."]
    assert prompt_brief.prompt_overrides(result) == {}


def test_no_key_is_said_in_a_few_words(folder, model):
    result = prompt_brief.read_prompt("A video to launch it", folder)
    assert len(model.calls) == 1 and result.source == "rules"
    assert "The model was not used (no valid model key is set); the prompt was read with built-in rules." in result.notes


@pytest.mark.parametrize("error,said", [
    (ModelProviderError("Gemini generate() failed: 429 RESOURCE_EXHAUSTED quota"), "its allowance is spent"),
    (ModelProviderError("blocked by the run's call policy: fallback to openrouter"), "the run's policy does not allow it"),
    (ModelProviderError("503 unavailable"), "it could not be asked (ModelProviderError)"),
    (ValueError("boom"), "it could not be asked (ValueError)")])
def test_a_failed_call_is_tried_once_and_named(error, said, folder, model):
    model.raises = error
    result = prompt_brief.read_prompt("A video to launch it", folder)
    assert len(model.calls) == 1                                           # no retry, even on a quota error
    assert result.source == "rules" and result.brief.objective == "launch"
    assert f"The model was not used ({said}); the prompt was read with built-in rules." in result.notes
    assert "503" not in " ".join(result.notes) and "RESOURCE_EXHAUSTED" not in " ".join(result.notes)


def test_a_model_that_does_not_answer_in_time_is_given_up_on(folder, model):
    answers(model, objective="enquiry")
    model.delay = 1.0
    started = time.monotonic()
    result = prompt_brief.read_prompt("A video to launch it", folder, timeout_s=0.1)

    assert time.monotonic() - started < 0.9
    assert result.source == "rules" and result.brief.objective == "launch"
    assert "The model was not used (it did not answer within 0.1 s); the prompt was read with built-in rules." \
        in result.notes


def test_the_call_runs_under_the_gateway_policy_without_touching_settings(folder, model, monkeypatch):
    answers(model)
    monkeypatch.setattr(prompt_brief.policy.settings, "openrouter_api_key", "kept", raising=False)
    prompt_brief.read_prompt("A calm reel", folder)

    (call_policy,) = model.policies
    assert call_policy is not None and "gemini" in call_policy.vendors and "openrouter" not in call_policy.vendors
    assert "creative_director" in call_policy.roles
    assert prompt_brief.policy.settings.openrouter_api_key == "kept"
    assert gateway._call_policy.get() is None and policy.active() is None


def test_the_policy_helper_limits_the_gateway_only_inside_its_block():
    with policy.model_call_policy() as call_policy:
        assert gateway._call_policy.get() is call_policy
        with pytest.raises(ModelProviderError, match="blocked by the run's call policy"):
            call_policy.check("blog_writer", "gemini")
        with pytest.raises(ModelProviderError, match="openrouter"):
            call_policy.check("creative_director", "openrouter", "fallback")
        call_policy.check("creative_director", "gemini")                   # allowed
    assert gateway._call_policy.get() is None
    assert policy.role_listed("creative_director") and not policy.role_listed("blog_writer")


def test_the_role_is_the_one_the_policy_lists_and_models_yaml_defines():
    from src.core.config import load_yaml_config

    assert prompt_brief.ROLE in load_yaml_config("video/policy.yaml")["model_roles"]
    assert prompt_brief.ROLE in load_yaml_config("models.yaml")["roles"]


# -- never set by a prompt -----------------------------------------------------------------------------------

def test_the_voice_the_picture_generated_frames_and_the_grade_are_never_set(folder, model):
    model.raises = None
    model.answer = {"voice": "male", "hero_image": "front.jpg", "allow_generated": True, "grade": False,
                    "duration_s": 60, "aspect": "16:9", "model": "seedance_2"}
    result = prompt_brief.read_prompt("A video. Use the male voice, open on front.jpg, allow generated frames.", folder)

    overrides = prompt_brief.prompt_overrides(result)
    for key in ("voice", "hero_image", "allow_generated", "grade", "model"):
        assert key not in overrides and key not in PromptBrief.model_fields
    assert result.brief.duration_s is None and result.brief.aspect is None      # length and shape: rules only
    assert PromptBrief(voice="male", allow_generated=True, grade=False).model_dump(exclude_none=True) == {}


def test_a_length_or_shape_in_the_models_answer_is_ignored_but_the_rules_still_read_them(folder, model):
    answers(model, duration_s=60, aspect="16:9")
    result = prompt_brief.read_prompt("A 15 second vertical reel", folder)
    assert (result.brief.duration_s, result.brief.aspect) == (15, "9:16")


# -- the cut and the overrides ---------------------------------------------------------------------------------

def test_a_prompt_over_4000_characters_is_cut_with_a_note(folder, model):
    answers(model)
    prompt = "A calm reel for families. " + "x" * 4500 + " ENDMARK launch"
    result = prompt_brief.read_prompt(prompt, folder)

    sent = model.calls[0][2]
    assert "ENDMARK" not in sent and "xxxx" in sent and sent.count("x") < 4001
    assert "The prompt is longer than 4,000 characters; only the first 4,000 were read." in result.notes
    assert result.brief.objective is None                                    # "launch" was past the cut
    assert "ENDMARK" not in " ".join(filter(None, [result.brief.creative_direction]))


def test_a_prompt_of_exactly_4000_characters_is_not_cut(folder):
    result = read("a" * 4000, folder)
    assert not [n for n in result.notes if "longer than" in n]


def test_prompt_overrides_has_only_the_keys_the_prompt_set(folder):
    result = read("A 10 second video, no captions", folder)
    assert prompt_brief.prompt_overrides(result) == {"duration_s": 10, "burn_captions": False}
    assert prompt_brief.prompt_overrides(read("make a video", folder)) == {}


def test_the_overrides_are_valid_options_that_lay_over_video_input_json(folder):
    (folder / "video_input.json").write_text(json.dumps({"duration_s": 15, "voice": "male", "cta": "Old"}), encoding="utf-8")
    result = read('A calm 30 second vertical video to launch it for young families. Highlight the pool. '
                  'End with "Book a site visit".', folder)
    overrides = prompt_brief.prompt_overrides(result)

    given = brief_agent.load_video_input(folder, overrides)
    assert isinstance(given, VideoInput)
    assert (given.duration_s, given.aspect, given.objective, given.cta) == (30, "9:16", "launch", "Book a site visit")
    assert given.highlights == ["Swimming pool"] and given.audience == "young families"
    assert given.voice == "male"                                           # the file's own, the prompt never sets it


def test_a_folder_with_no_property_json_is_an_ingestion_error(tmp_path):
    with pytest.raises(IngestionError):
        read("A video", tmp_path)


# -- the three kinds of prompt ------------------------------------------------------------------------------------

def test_a_medeo_style_prompt_is_mostly_not_used_and_says_so(folder):
    prompt = (
        "Cinematic drone shot flying forward over a luxurious high-rise residential complex, golden hour lighting, "
        "lush green trees swaying gently, photorealistic, ultra-detailed 8k resolution.\n"
        "Motion Controls: Set the camera motion to Dolly In or Slow Zoom In. Keep the motion scale moderate.\n"
        'Text Overlay (Add in post): "Elevate Your Lifestyle." Prompt: Smooth slow pan across landscaped gardens.\n'
        "Motion Controls: Set the camera to Pan Right. This creates a sweeping reveal of the property.\n"
        'Text Overlay (Add in post): "Brigade Group. Building Positive Experiences." (Add a fade-out to the logo).')
    result = read(prompt, folder)

    assert prompt_brief.prompt_overrides(result) == {"tone": "refined"}        # "luxurious" is all it states
    families = " ".join(result.notes)
    for expected in ("Camera moves in the prompt are not used", "dolly", "Resolution and render-quality words",
                     "8k", "Scene descriptions and on-screen text are not used", "Quoted lines are not printed (2)"):
        assert expected in families, expected


def test_a_made_up_amenity_and_price_are_refused(folder):
    result = read('A 45 sec teaser to launch it. Highlight the rooftop helipad, the pool and the gym. '
                  'Priced at Rs 49 lakh, 20% off this week. CTA: "Call 99999 00000 now".', folder)

    assert prompt_brief.prompt_overrides(result) == {
        "duration_s": 45, "objective": "launch", "highlights": ["Swimming pool", "Gymnasium"]}
    assert "'rooftop helipad' is not in this property's details; left out." in result.notes
    assert any("Rs 49 lakh" in n for n in result.notes)
    assert any("The call to action has a figure (00000)" in n for n in result.notes)
