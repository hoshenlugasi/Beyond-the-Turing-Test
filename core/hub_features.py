"""Explicit nullable loading types; no datasets/Arrow dependency is needed."""
from core.event_schema import SCHEMAS


def feature(shape):
    if isinstance(shape, dict):
        return {key: feature(value) for key, value in shape.items()}
    if isinstance(shape, list):
        return {"feature": feature(shape[0]), "_type": "List"}
    return {"dtype": {"str": "string", "int": "int64", "float": "float64",
                       "bool": "bool"}[shape], "_type": "Value"}


def dataset_infos():
    """Loading metadata for all five views, including all-null shard fields."""
    fields = {
        "games": dict(game_id="str", ai_pseudonym="str", num_human_players="int",
                      total_rounds="int", ai_eliminated="bool", ai_elimination_round="int",
                      event_count="int", human_start_count="int", path="str"),
        "messages": dict(game_id="str", message_order="int", recorded_time="str",
                         speaker="str", speaker_role="str", text="str",
                         elapsed_seconds="float", time_basis="str"),
        "votes": dict(game_id="str", round="int", voter="str", voted_for="str",
                      is_ai_voter="bool"),
        "surveys": dict(game_id="str", participant="str", survey_schema="str",
                        human_likeness="int", timing="int", relevance="int",
                        enjoyment="int", comments="str", suspicion_reason="str",
                        additional_comments="str"),
        "events": dict(event_schema="int", game_id="str", event_id="str",
                       event_order="int", component="str", action="str", turn_id="str",
                       elapsed_seconds="float", time_basis="str",
                       time_precision_seconds="float", event_text="str", payload=SCHEMAS),
    }
    return {name: {"config_name": name, "features": feature(shape)}
            for name, shape in fields.items()}