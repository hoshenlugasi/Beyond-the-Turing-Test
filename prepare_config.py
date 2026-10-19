"""
usage: prepare_config.py [-h] [-o OUTPUT] [-p PLAYERS] [-m MAFIA] [-l {0,1}]
                         [-b] [-n NAMES_FILE] [-c] [-j LLM_CONFIG_JSON_PATH]
                         [-dt DAYTIME_MINUTES] [-nt NIGHTTIME_MINUTES]

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output config file name
  -p PLAYERS, --players PLAYERS
                        total number of players in game
  -m MAFIA, --mafia MAFIA
                        number of mafia players in game
  -l {0,1}, --llm {0,1}
                        number of LLM players in game, currently supports maximum 1
  -b, --bystander       whether the LLM player can only be bystander (not mafia)
  -n NAMES_FILE, --names_file NAMES_FILE
                        path to file with the participating players' real
                        names (before code names assignment), separated by new
                        line breaks ('\\n')
  -c, --change_llm_config
                        whether to edit the default LLM configuration
  -j LLM_CONFIG_JSON_PATH, --llm_config_json_path LLM_CONFIG_JSON_PATH
                        optional path to LLM configuration as json (has to be complete)
  -dt DAYTIME_MINUTES, --daytime_minutes DAYTIME_MINUTES
                        number of minutes for Daytime phase
  -nt NIGHTTIME_MINUTES, --nighttime_minutes NIGHTTIME_MINUTES
                        number of minutes for Nighttime phase

Process finished with exit code 0

"""
import json
import argparse
import random
import sys
import time
from pathlib import Path
from termcolor import colored
from dataclasses import dataclass, asdict, field
from core.config_schema import reject_retired_config
from core.game_constants import DEFAULT_CONFIG_DIR, DEFAULT_NUM_PLAYERS, OPTIONAL_CODE_NAMES, \
    PLAYERS_KEY_IN_CONFIG, DEFAULT_DAYTIME_MINUTES, \
    DAYTIME_MINUTES_KEY, ANONYMOUS_VOTING_KEY
from llm_players.llm_constants import INT_CONFIG_KEYS, FLOAT_CONFIG_KEYS, DEFAULT_LLM_CONFIG, \
    LLM_CONFIG_KEYS_OPTIONS, BOOL_CONFIG_KEYS, PERSONA_KEY, DEFAULT_PERSONA_ID, load_personas, \
    ASYNC_TYPE_KEY, SCHEDULE_THEN_GENERATE_TYPE, REVIEWED_SCHEDULE_THEN_GENERATE_TYPE, \
    MODEL_NAME_KEY, USE_GEMINI_KEY, USE_TOGETHER_KEY, USE_OPENROUTER_KEY

LLM_CONFIG_KEYS_INDEXED_OPTIONS = {
    key: {f"{i}": option for (i, option) in enumerate(options)}
    for key, options in LLM_CONFIG_KEYS_OPTIONS.items()
}

# Local CLI defaults only; scientific prompts and shared model constants stay intact.
LOCAL_AGENT_DEFAULTS = {
    "model_name": "gemini-2.5-flash",
    "use_gemini": True,
    "use_together": False,
    "use_openrouter": False,
    "use_pipeline": False,
    "async_type": "social_turing_test_agent",
    "scheduler_model": "gemini-2.5-flash",
    "reviewer_enabled": True,
    "reviewer_backend": "gemini",
    "reviewer_model": "gemini-2.5-flash",
    "multi_candidate_generation": True,
    "num_candidates": 4,
    "investigator": True,
    "voting_strategy": "strategic",
    "pipeline_timeout_s": 35.0,
    "orchestration_mode": "synchronous",
    "allow_active_authoritative": False,
    "persona": "swiss_student",
    "agent_code": "current",
}


@dataclass
class PlayerConfig:
    name: str  # player's code name from the game's pool (in constants file)
    is_ai: bool = False  # is this player the AI (imposter)?
    is_llm: bool = False  # is this player controlled by LLM?
    real_name: str = ""
    llm_config: dict = field(default_factory=dict)
    
    # Backwards compatibility property
    @property
    def is_mafia(self):
        """Backwards compatibility: is_mafia maps to is_ai"""
        return self.is_ai
    
    @is_mafia.setter
    def is_mafia(self, value):
        """Backwards compatibility: setting is_mafia sets is_ai"""
        self.is_ai = value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default=None, help="output config file name")
    parser.add_argument("-p", "--players", type=int, default=None,
                        help="total number of players in game")
    parser.add_argument("-m", "--mafia", type=int, default=None,
                        help="number of mafia players in game")
    parser.add_argument("-l", "--llm", type=int, default=1, choices=[0, 1],
                        # since participants will rank the (single) LLM player performance
                        help="number of LLM players in game, currently supports maximum 1")
    parser.add_argument("-b", "--bystander", action="store_true",
                        help="whether the LLM player can only be bystander (not mafia)")
    parser.add_argument("-n", "--names_file", default=None,
                        help="path to file with the participating players' real names "
                             "(before code names assignment), separated by new line breaks ('\\n')")
    parser.add_argument("-c", "--change_llm_config", action="store_true",
                        help="whether to edit the default LLM configuration")
    parser.add_argument("-j", "--llm_config_json_path", default=None,
                        help="optional path to LLM configuration as json (has to be complete)")
    parser.add_argument("-dt", "--daytime_minutes", type=float, default=DEFAULT_DAYTIME_MINUTES,
                        help="number of minutes for Daytime phase")
    parser.add_argument("-nt", "--nighttime_minutes", type=float, default=0,
                        help="number of minutes for Nighttime phase (deprecated - always 0 in Social Turing Test)")
    parser.add_argument("-a", "--anonymous_voting", action="store_true",
                        help="whether to use anonymous voting (show vote counts instead of individual votes)")
    parser.add_argument("-pe", "--persona", default=None,
                        help="persona for the AI player (e.g., 'introverted_teen', 'sarcastic_player', etc.)")
    parser.add_argument("--event-context", default=None,
                        help="shared real-world context ALL human players are in (e.g. "
                             "'the IML course hackathon at the Hebrew University, sitting in rooms a/b/c'). "
                             "Injected into the AI's prompts so private shared context can't expose it.")
    parser.add_argument("-t", "--topic", default=None,
                        help="conversation topic for the game (e.g., 'favorite_movies', 'hobbies', etc.)")
    parser.add_argument("-r", "--review-mode",
                        choices=["none", "gemini", "together", "openrouter"],
                        default=None,
                        help="reviewer backend: none, gemini, together, or openrouter")
    parser.add_argument("--generator-model", default=None,
                        help="override generator model name (e.g., 'gemini-2.5-flash', "
                             "'meta-llama/Llama-3.3-70B-Instruct-Turbo', 'openai/gpt-4o-mini')")
    parser.add_argument("--reviewer-model", default=None,
                        help="override reviewer model name (requires --review-mode)")
    parser.add_argument("--scheduler-model", default=None,
                        help="Override the send/wait scheduler model so it can "
                             "differ from the generator model (provider auto-detected). "
                            "Default: gemini-2.5-flash.")
    parser.add_argument("--provider",
                        choices=["auto", "gemini", "together", "openrouter", "huggingface"],
                        default="auto",
                        help="generator provider override. 'auto' (default) infers from model name.")
    parser.add_argument("--test", action="store_true",
                        help="prepare this as a test game under games/test/. "
                             "(short -t is taken by --topic, so this flag is long-form only.)")
    args = parser.parse_args()
    return args


# Prefixes that strongly indicate an OpenRouter-style 'vendor/model' slug.
# These are vendors available ONLY through OpenRouter (not via Together or Vertex).
_OPENROUTER_VENDOR_PREFIXES = (
    "openai/", "anthropic/", "x-ai/", "xai/", "perplexity/",
    "cohere/", "mistralai/", "deepseek/", "nousresearch/",
)
_OPENROUTER_NAME_HINTS = ("gpt-", "claude-", "grok-", "o1-", "o3-", "o4-")


def detect_provider_from_model(model_name):
    """Infer provider from a model-name string. Used when --provider=auto.

    Order of checks:
      1. Gemini/Google models -> Vertex AI
      2. Known OpenRouter-only vendor prefixes or name hints -> OpenRouter
      3. Default: Together AI
    """
    if not model_name:
        return "together"
    n = model_name.lower()
    if "gemini" in n or n.startswith("google/"):
        return "gemini"
    if any(n.startswith(p) for p in _OPENROUTER_VENDOR_PREFIXES):
        return "openrouter"
    if any(h in n for h in _OPENROUTER_NAME_HINTS):
        return "openrouter"
    return "together"


def _apply_provider_flags(llm_config, provider):
    """Set mutually-exclusive provider flags in llm_config."""
    llm_config[USE_GEMINI_KEY] = (provider == "gemini")
    llm_config[USE_TOGETHER_KEY] = (provider == "together")
    llm_config[USE_OPENROUTER_KEY] = (provider == "openrouter")
    # Pipeline (local HF) is handled separately; keep existing value unless hf chosen
    if provider == "huggingface":
        llm_config["use_pipeline"] = True
    else:
        llm_config["use_pipeline"] = False


def handle_output_file(args):
    output_file = args.output
    if output_file is None:
        output_file = "config" + time.strftime("%d%m%y_%H%M")
    if not output_file.startswith(DEFAULT_CONFIG_DIR):
        output_file = DEFAULT_CONFIG_DIR + "/" + output_file
    if not output_file.endswith(".json"):
        output_file += ".json"
    output_file = Path(output_file)
    if output_file.exists():
        input("Warning: the destination output path already exists... "
              "[enter to continue and override, Ctrl+C to cancel]")
    else:
        output_file.touch()  # to raise an error immediately if the new path is invalid
    return output_file


def validate_names_file(args):
    if args.names_file is not None and not Path(args.names_file).exists():
        raise ValueError("Provided names file (with -n/--names_file) does not exist:",
                         args.names_file)


def handle_num_players(args):
    num_players = args.players
    if num_players is None:
        num_players = DEFAULT_NUM_PLAYERS
        print(f"Using default number of total players: {DEFAULT_NUM_PLAYERS}")
    elif num_players < 3:
        raise ValueError(f"{num_players} is not enough players for Social Turing Test! "
                         f"Minimum is 3 players (1 AI + 2 humans).")
    else:
        print(f"Using number of total players: {num_players}")
    
    # Social Turing Test: Always exactly 1 imposter (will be the AI)
    num_mafia = 1
    print(f"Social Turing Test mode: Exactly 1 Imposter (AI) among {num_players} total players")
    
    code_names = random.sample(OPTIONAL_CODE_NAMES, min(num_players, len(OPTIONAL_CODE_NAMES)))
    used = set(code_names)
    suffix = 1
    while len(code_names) < num_players:
        codename = f"Player{suffix}"
        suffix += 1
        if codename not in used:
            code_names.append(codename)
            used.add(codename)
    player_configs = [PlayerConfig(code_name) for code_name in code_names]
    
    # Select one player to be the imposter (will be assigned to AI)
    mafia_player = random.choice(player_configs)
    mafia_player.is_mafia = True
    
    return player_configs


def get_llm_config(llm_numbered_symbol, args):
    llm_config = DEFAULT_LLM_CONFIG.copy()
    llm_config.update(LOCAL_AGENT_DEFAULTS)
    if args.llm_config_json_path is not None:
        print("Using the LLM configuration in provided path:", args.llm_config_json_path)
        with open(args.llm_config_json_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        reject_retired_config(overrides)
        llm_config.update(overrides)
    
    # Apply --generator-model override
    if args.generator_model is not None:
        llm_config[MODEL_NAME_KEY] = args.generator_model
        provider = args.provider if args.provider != "auto" else detect_provider_from_model(args.generator_model)
        _apply_provider_flags(llm_config, provider)
        label = {"gemini": "Vertex AI", "together": "Together AI",
                 "openrouter": "OpenRouter", "huggingface": "HuggingFace (local)"}[provider]
        print(colored(f"✓ Generator model: {args.generator_model} ({label})", "green"))
    elif args.provider != "auto":
        # --provider passed without --generator-model: override flags on default model
        _apply_provider_flags(llm_config, args.provider)
        print(colored(f"✓ Provider override: {args.provider}", "green"))

    # Apply --scheduler-model override: dedicated send/wait model.
    if args.scheduler_model is not None and str(args.scheduler_model).strip():
        llm_config["scheduler_model"] = str(args.scheduler_model).strip()
        sched_provider = detect_provider_from_model(args.scheduler_model)
        print(colored(f"✓ Scheduler model: {args.scheduler_model} ({sched_provider})", "green"))

    # Shared event context (e.g. a physical hackathon all players attend) — the
    # AI's only defence against private-shared-context traps ("where are we?").
    if getattr(args, "event_context", None):
        llm_config["event_context"] = str(args.event_context).strip()
        print(colored(f"✓ Event context: {llm_config['event_context']}", "green"))

    # Handle persona selection with improved validation
    if args.persona is not None:
        personas = load_personas()
        if args.persona in personas:
            llm_config[PERSONA_KEY] = args.persona
            persona_info = personas[args.persona]
            persona_display_name = persona_info.get('name', args.persona)
            print(colored(f"✓ Using persona: {persona_display_name} ('{args.persona}')", "green"))
        else:
            print(colored(f"✗ Error: Persona '{args.persona}' not found!", "red"))
            available = list(personas.keys()) if personas else ['israeli_student', 'american_student', 'british_student', 'australian_student']
            print(f"Available personas: {', '.join(sorted(available))}")
            print(colored(f"Using default persona instead.", "yellow"))
            llm_config[PERSONA_KEY] = DEFAULT_PERSONA_ID
    elif PERSONA_KEY not in llm_config:
        # Ensure default persona is set if not specified
        llm_config[PERSONA_KEY] = DEFAULT_PERSONA_ID
    
    if args.change_llm_config:
        config_approved = False
        index2key = {f"{i}": key for i, key in enumerate(llm_config.keys())}
        while not config_approved:
            print(f"Here is the current config for {llm_numbered_symbol}:")
            for i, key in index2key.items():
                print(f"{i}.\t{key}: {llm_config[key]}")
            index = input("Enter a key index to change its value, "
                          "or anything else to approve the config: ")
            if index in index2key:
                key = index2key[index]
                if key in INT_CONFIG_KEYS:
                    llm_config[key] = int(input(f"Enter a new number for {key}: "))
                elif key in FLOAT_CONFIG_KEYS:
                    llm_config[key] = float(input(f"Enter a new value for {key}: "))
                elif key in BOOL_CONFIG_KEYS:
                    llm_config[key] = eval(input(f"Enter True/False for {key}: ").capitalize())
                else:
                    # Handle persona and other option-based configs
                    if key == PERSONA_KEY:
                        personas = load_personas()
                        options = list(personas.keys()) if personas else ['israeli_student', 'american_student', 'british_student', 'australian_student']
                        indexed_options = {f"{i}": option for i, option in enumerate(options)}
                    elif callable(LLM_CONFIG_KEYS_OPTIONS.get(key)):
                        options = LLM_CONFIG_KEYS_OPTIONS[key]()
                        indexed_options = {f"{i}": option for i, option in enumerate(options)}
                    else:
                        indexed_options = LLM_CONFIG_KEYS_INDEXED_OPTIONS.get(key, {})
                    
                    choice = None
                    while choice not in indexed_options:
                        all_options = "\n".join([f"\t{i}: {option}" for (i, option)
                                                 in indexed_options.items()])
                        choice = input(f"Choose the wanted option for {key}:\n"
                                       f"{all_options}\n")
                    llm_config[key] = indexed_options[choice]
            else:
                config_approved = True
    # Apply --review-mode if specified
    if args.review_mode is not None:
        llm_config["reviewer_enabled"] = args.review_mode != "none"
        if args.review_mode == "none":
            llm_config[ASYNC_TYPE_KEY] = SCHEDULE_THEN_GENERATE_TYPE
            llm_config.pop("reviewer_backend", None)
            llm_config.pop("reviewer_model", None)
            print(colored("Reviewer: disabled", "yellow"))
        elif args.review_mode == "gemini":
            llm_config[ASYNC_TYPE_KEY] = REVIEWED_SCHEDULE_THEN_GENERATE_TYPE
            llm_config["reviewer_backend"] = "gemini"
            llm_config["reviewer_model"] = "gemini-2.5-flash"
            print(colored("Reviewer: Gemini (gemini-2.5-flash)", "green"))
        elif args.review_mode == "together":
            llm_config[ASYNC_TYPE_KEY] = REVIEWED_SCHEDULE_THEN_GENERATE_TYPE
            llm_config["reviewer_backend"] = "together"
            llm_config["reviewer_model"] = "Qwen/Qwen3-Next-80B-A3B-Instruct"
            print(colored("Reviewer: Together AI (Qwen3-Next-80B)", "green"))
        elif args.review_mode == "openrouter":
            llm_config[ASYNC_TYPE_KEY] = REVIEWED_SCHEDULE_THEN_GENERATE_TYPE
            llm_config["reviewer_backend"] = "openrouter"
            llm_config["reviewer_model"] = "openai/gpt-4o-mini"
            print(colored("Reviewer: OpenRouter (openai/gpt-4o-mini)", "green"))
        # Apply --reviewer-model override
        if args.reviewer_model is not None and args.review_mode != "none":
            llm_config["reviewer_model"] = args.reviewer_model
            print(colored(f"✓ Reviewer model override: {args.reviewer_model}", "green"))
    elif args.reviewer_model is not None:
        print(colored("⚠ --reviewer-model ignored without --review-mode", "yellow"))

    # Reviewer options configure roles within the single agent.
    llm_config[ASYNC_TYPE_KEY] = "social_turing_test_agent"
    llm_config["orchestration_mode"] = "synchronous"
    llm_config["allow_active_authoritative"] = False
    llm_config["agent_code"] = "current"

    reject_retired_config(llm_config)
    return llm_config


def handle_llm_participation(args, player_configs):
    num_llms = args.llm
    print(f"Using {num_llms} LLM player{'' if num_llms == 1 else ''}")
    if num_llms > 0:
        # Social Turing Test: AI is ALWAYS the Imposter
        # Find the player that was assigned is_mafia=True
        mafia_player = [player for player in player_configs if player.is_mafia][0]
        
        # Assign the LLM to be the Imposter
        mafia_player.is_llm = True
        mafia_player.real_name = "LLM0"
        mafia_player.llm_config = get_llm_config(mafia_player.real_name, args)
        
        print(f"Social Turing Test: The AI is assigned as the Imposter")


def assign_real_names(args, player_configs):
    human_players = [player_config for player_config in player_configs if not player_config.is_llm]
    if args.names_file is None:
        real_names = []
    else:  # split() removes the '\n' and ignores empty lines
        real_names = Path(args.names_file).read_text().split()
    if len(set(real_names)) < len(real_names):
        raise ValueError("The provided names file contain reoccurrence of the same name!")
    if real_names:
        print("The given real names of participating players:")
        print(",  ".join([f"{i + 1}. {name}" for i, name in enumerate(real_names)]))
    if len(human_players) < len(real_names):
        print(f"Warning: {len(real_names)} participating names given, "
              f"only {len(human_players)} needed for human players, "
              f"using only the first {len(human_players)}...")
        real_names = real_names[:len(human_players)]
    elif len(human_players) > len(real_names):
        while len(human_players) > len(real_names):
            new_real_name = input("Still not enough real names of participating players, "
                                  "provide another one: ").strip()
            if not new_real_name or new_real_name in real_names:
                print("This name belongs to a player that was already listed!")
            else:
                real_names.append(new_real_name)
    # now len(human_players) == len(real_names)
    for human_player, real_name in zip(human_players, real_names):
        human_player.real_name = real_name


def save_config(args, output_file, player_configs):
    # Social Turing Test: no night phase, so nighttime = 0
    config = {PLAYERS_KEY_IN_CONFIG: [asdict(player_config) for player_config in player_configs],
              DAYTIME_MINUTES_KEY: args.daytime_minutes,
              "nighttime_minutes": 0,  # Always 0 for Social Turing Test (no night phase)
              ANONYMOUS_VOTING_KEY: args.anonymous_voting,
              "is_test": bool(getattr(args, "test", False)),
              "notes": input("Add notes to this config: [or enter to skip] ").strip(),
              "preparation_command": " ".join(sys.argv)}
    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)
    print("Configuration was created and saved to:", colored(output_file, "green"))


def main():
    args = parse_args()
    output_file = handle_output_file(args)  # done first to immediately raise error if invalid
    validate_names_file(args)
    player_configs = handle_num_players(args)
    handle_llm_participation(args, player_configs)
    assign_real_names(args, player_configs)
    save_config(args, output_file, player_configs)


if __name__ == '__main__':
    main()
