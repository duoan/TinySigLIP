"""
Evaluation script for TinySigLIP zero-shot classification on ImageNet-1k.
Similar to TinyCLIP's evaluation approach.

This script evaluates the model on ImageNet-1k zero-shot classification task.
For image-text retrieval evaluation, use eval_retrieval.py instead.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageNet
from tqdm import tqdm
from transformers import AutoTokenizer

from tinysiglip.model import TinySiglipModel
from tinysiglip.processor import TinySiglipProcessor

# ImageNet-1k class names (1000 classes)
IMAGENET_CLASSES = [
    "tench",
    "goldfish",
    "great white shark",
    "tiger shark",
    "hammerhead shark",
    "electric ray",
    "stingray",
    "cock",
    "hen",
    "ostrich",
    "brambling",
    "goldfinch",
    "house finch",
    "junco",
    "indigo bunting",
    "robin",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "water ouzel",
    "kite",
    "bald eagle",
    "vulture",
    "great grey owl",
    "European fire salamander",
    "common newt",
    "eft",
    "spotted salamander",
    "axolotl",
    "bullfrog",
    "tree frog",
    "tailed frog",
    "loggerhead",
    "leatherback turtle",
    "mud turtle",
    "terrapin",
    "turtle",
    "banded gecko",
    "common iguana",
    "American chameleon",
    "whiptail",
    "agama",
    "frilled lizard",
    "alligator lizard",
    "Gila monster",
    "green lizard",
    "African chameleon",
    "Komodo dragon",
    "African crocodile",
    "American alligator",
    "triceratops",
    "thunder snake",
    "ringneck snake",
    "hognose snake",
    "green snake",
    "king snake",
    "garter snake",
    "water snake",
    "vine snake",
    "night snake",
    "boa constrictor",
    "rock python",
    "Indian cobra",
    "green mamba",
    "sea snake",
    "horned viper",
    "diamondback",
    "sidewinder",
    "trilobite",
    "harvestman",
    "scorpion",
    "black and gold garden spider",
    "barn spider",
    "garden spider",
    "black widow",
    "tarantula",
    "wolf spider",
    "tick",
    "centipede",
    "black grouse",
    "ptarmigan",
    "ruffed grouse",
    "prairie chicken",
    "peacock",
    "quail",
    "partridge",
    "African grey",
    "macaw",
    "sulphur-crested cockatoo",
    "lorikeet",
    "coucal",
    "bee eater",
    "hornbill",
    "hummingbird",
    "jacamar",
    "toucan",
    "drake",
    "red-breasted merganser",
    "goose",
    "black swan",
    "tusker",
    "echidna",
    "platypus",
    "wallaby",
    "koala",
    "wombat",
    "jellyfish",
    "sea anemone",
    "brain coral",
    "flatworm",
    "nematode",
    "conch",
    "snail",
    "slug",
    "sea slug",
    "chiton",
    "chambered nautilus",
    "king crab",
    "American lobster",
    "spiny lobster",
    "crayfish",
    "hermit crab",
    "isopod",
    "white stork",
    "black stork",
    "spoonbill",
    "flamingo",
    "little blue heron",
    "American egret",
    "bittern",
    "crane",
    "limpkin",
    "European gallinule",
    "American coot",
    "bustard",
    "ruddy turnstone",
    "red-backed sandpiper",
    "redshank",
    "dowitcher",
    "oystercatcher",
    "pelican",
    "king penguin",
    "albatross",
    "grey whale",
    "killer whale",
    "dugong",
    "sea lion",
    "Chihuahua",
    "Japanese spaniel",
    "Maltese dog",
    "Pekinese",
    "Shih-Tzu",
    "Blenheim spaniel",
    "papillon",
    "toy terrier",
    "Rhodesian ridgeback",
    "Afghan hound",
    "basset",
    "beagle",
    "bloodhound",
    "bluetick",
    "black-and-tan coonhound",
    "Walker hound",
    "English foxhound",
    "redbone",
    "borzoi",
    "Irish wolfhound",
    "Italian greyhound",
    "whippet",
    "Ibizan hound",
    "Norwegian elkhound",
    "otterhound",
    "Saluki",
    "Scottish deerhound",
    "Weimaraner",
    "Staffordshire bullterrier",
    "American Staffordshire terrier",
    "Bedlington terrier",
    "Border terrier",
    "Kerry blue terrier",
    "Irish terrier",
    "Norfolk terrier",
    "Norwich terrier",
    "Yorkshire terrier",
    "wire-haired fox terrier",
    "Lakeland terrier",
    "Sealyham terrier",
    "Airedale",
    "cairn",
    "Australian terrier",
    "Dandie Dinmont",
    "Boston bull",
    "miniature schnauzer",
    "giant schnauzer",
    "standard schnauzer",
    "Scotch terrier",
    "Tibetan terrier",
    "silky terrier",
    "soft-coated wheaten terrier",
    "West Highland white terrier",
    "Lhasa",
    "flat-coated retriever",
    "curly-coated retriever",
    "golden retriever",
    "Labrador retriever",
    "Chesapeake Bay retriever",
    "German short-haired pointer",
    "vizsla",
    "English setter",
    "Irish setter",
    "Gordon setter",
    "Brittany spaniel",
    "clumber",
    "English springer",
    "Welsh springer spaniel",
    "cocker spaniel",
    "Sussex spaniel",
    "Irish water spaniel",
    "kuvasz",
    "schipperke",
    "groenendael",
    "malinois",
    "briard",
    "kelpie",
    "komondor",
    "Old English sheepdog",
    "Shetland sheepdog",
    "collie",
    "Border collie",
    "Bouvier des Flandres",
    "Rottweiler",
    "German shepherd",
    "Doberman",
    "miniature pinscher",
    "Greater Swiss Mountain dog",
    "Bernese mountain dog",
    "Appenzeller",
    "EntleBucher",
    "boxer",
    "bull mastiff",
    "Tibetan mastiff",
    "French bulldog",
    "Great Dane",
    "Saint Bernard",
    "Eskimo dog",
    "malamute",
    "Siberian husky",
    "affenpinscher",
    "basenji",
    "pug",
    "Leonberg",
    "Newfoundland",
    "Great Pyrenees",
    "Samoyed",
    "Pomeranian",
    "chow",
    "keeshond",
    "Brabancon griffon",
    "Pembroke",
    "Cardigan",
    "toy poodle",
    "miniature poodle",
    "standard poodle",
    "Mexican hairless",
    "timber wolf",
    "white wolf",
    "red wolf",
    "coyote",
    "dingo",
    "dhole",
    "African hunting dog",
    "hyena",
    "red fox",
    "kit fox",
    "Arctic fox",
    "grey fox",
    "tabby",
    "tiger cat",
    "Persian cat",
    "Siamese cat",
    "Egyptian cat",
    "lion",
    "tiger",
    "jaguar",
    "leopard",
    "snow leopard",
    "lynx",
    "bobcat",
    "clouded leopard",
    "clouded leopard",
    "leopard cat",
    "cheetah",
    "brown bear",
    "American black bear",
    "ice bear",
    "sloth bear",
    "mongoose",
    "meerkat",
    "tiger beetle",
    "ladybug",
    "ground beetle",
    "long-horned beetle",
    "leaf beetle",
    "dung beetle",
    "rhinoceros beetle",
    "weevil",
    "fly",
    "bee",
    "ant",
    "grasshopper",
    "cricket",
    "walking stick",
    "cockroach",
    "mantis",
    "cicada",
    "leafhopper",
    "lacewing",
    "dragonfly",
    "damselfly",
    "admiral",
    "ringlet",
    "monarch",
    "cabbage butterfly",
    "sulphur butterfly",
    "lycaenid",
    "starfish",
    "sea urchin",
    "sea cucumber",
    "wood rabbit",
    "hare",
    "Angora",
    "hamster",
    "porcupine",
    "fox squirrel",
    "marmot",
    "beaver",
    "guinea pig",
    "sorrel",
    "zebra",
    "hog",
    "wild boar",
    "warthog",
    "hippopotamus",
    "ox",
    "water buffalo",
    "bison",
    "ram",
    "bighorn",
    "ibex",
    "hartebeest",
    "impala",
    "gazelle",
    "Arabian camel",
    "llama",
    "weasel",
    "mink",
    "polecat",
    "black-footed ferret",
    "otter",
    "skunk",
    "badger",
    "armadillo",
    "three-toed sloth",
    "orangutan",
    "gorilla",
    "chimpanzee",
    "gibbon",
    "siamang",
    "guenon",
    "patas",
    "baboon",
    "macaque",
    "langur",
    "colobus",
    "proboscis monkey",
    "marmoset",
    "capuchin",
    "howler monkey",
    "titi",
    "spider monkey",
    "squirrel monkey",
    "Madagascar cat",
    "indri",
    "Indian elephant",
    "African elephant",
    "lesser panda",
    "giant panda",
    "barracouta",
    "eel",
    "coho",
    "rock beauty",
    "anemone fish",
    "sturgeon",
    "gar",
    "lionfish",
    "puffer",
    "abacus",
    "academic gown",
    "accordion",
    "acoustic guitar",
    "aircraft carrier",
    "airliner",
    "airship",
    "altar",
    "ambulance",
    "amphibian",
    "analog clock",
    "apiary",
    "apron",
    "ashcan",
    "assault rifle",
    "backpack",
    "bakery",
    "balance beam",
    "balloon",
    "ballpoint",
    "Band Aid",
    "banjo",
    "bannister",
    "barbell",
    "barber chair",
    "barbershop",
    "barn",
    "barometer",
    "barrel",
    "barrow",
    "baseball",
    "basketball",
    "bassinet",
    "bassoon",
    "bathing cap",
    "bath towel",
    "bathtub",
    "beach wagon",
    "beacon",
    "beaker",
    "bearskin",
    "beer bottle",
    "beer glass",
    "bell cote",
    "bib",
    "bicycle-built-for-two",
    "bikini",
    "binder",
    "binoculars",
    "birdhouse",
    "boathouse",
    "bobsled",
    "bolo tie",
    "bonnet",
    "bookcase",
    "bookshop",
    "bottlecap",
    "bow",
    "bow tie",
    "brass",
    "brassiere",
    "breakwater",
    "breastplate",
    "broom",
    "bucket",
    "buckle",
    "bulletproof vest",
    "bullet train",
    "butcher shop",
    "cab",
    "caldron",
    "candle",
    "cannon",
    "canoe",
    "can opener",
    "cardigan",
    "car mirror",
    "carousel",
    "carpenter's kit",
    "carton",
    "car wheel",
    "cash machine",
    "cassette",
    "cassette player",
    "castle",
    "catamaran",
    "CD player",
    "cello",
    "cellular telephone",
    "chain",
    "chainlink fence",
    "chain mail",
    "chain saw",
    "chest",
    "chiffonier",
    "chime",
    "china cabinet",
    "Christmas stocking",
    "church",
    "cinema",
    "cleaver",
    "cliff dwelling",
    "cloak",
    "clog",
    "cocktail shaker",
    "coffee mug",
    "coffeepot",
    "coil",
    "combination lock",
    "computer keyboard",
    "confectionery",
    "container ship",
    "convertible",
    "corkscrew",
    "cornet",
    "cowboy boot",
    "cowboy hat",
    "cradle",
    "crane",
    "crash helmet",
    "crate",
    "crib",
    "Crock Pot",
    "croquet ball",
    "crutch",
    "cuirass",
    "dam",
    "desk",
    "desktop computer",
    "dial telephone",
    "diaper",
    "digital clock",
    "digital watch",
    "dining table",
    "dishrag",
    "dishwasher",
    "disk brake",
    "dock",
    "dogsled",
    "dome",
    "doormat",
    "drilling platform",
    "drum",
    "drumstick",
    "dumbbell",
    "Dutch oven",
    "electric fan",
    "electric guitar",
    "electric locomotive",
    "entertainment center",
    "envelope",
    "espresso maker",
    "face powder",
    "feather boa",
    "file",
    "fireboat",
    "fire engine",
    "fire screen",
    "flagpole",
    "flute",
    "folding chair",
    "football helmet",
    "forklift",
    "fountain",
    "fountain pen",
    "four-poster",
    "freight car",
    "French horn",
    "frying pan",
    "fur coat",
    "garbage truck",
    "garden cart",
    "gasmask",
    "gas pump",
    "goblet",
    "go-kart",
    "golf ball",
    "golfcart",
    "gondola",
    "gong",
    "gown",
    "grand piano",
    "greenhouse",
    "grille",
    "grocery store",
    "guillotine",
    "hair slide",
    "hair spray",
    "half track",
    "hammer",
    "hamper",
    "hand blower",
    "hand-held computer",
    "handkerchief",
    "hard disc",
    "harmonica",
    "harp",
    "harvester",
    "hatchet",
    "holster",
    "home theater",
    "honeycomb",
    "hook",
    "hoopskirt",
    "horizontal bar",
    "horse cart",
    "hourglass",
    "iPod",
    "iron",
    "jack-o'-lantern",
    "jean",
    "jeep",
    "jersey",
    "jigsaw puzzle",
    "jinrikisha",
    "joystick",
    "kimono",
    "knee pad",
    "knot",
    "lab coat",
    "ladle",
    "lampshade",
    "laptop",
    "lawn mower",
    "lens cap",
    "letter opener",
    "library",
    "lifeboat",
    "lighter",
    "limousine",
    "liner",
    "lipstick",
    "Loafer",
    "lotion",
    "loudspeaker",
    "loupe",
    "lumbermill",
    "magnetic compass",
    "mailbag",
    "mailbox",
    "maillot",
    "maillot tank suit",
    "manhole cover",
    "maraca",
    "marimba",
    "mask",
    "matchstick",
    "maypole",
    "maze",
    "measuring cup",
    "medicine chest",
    "megalith",
    "microphone",
    "microwave",
    "military uniform",
    "milk can",
    "minibus",
    "miniskirt",
    "minivan",
    "missile",
    "mitten",
    "mixing bowl",
    "mobile home",
    "Model T",
    "modem",
    "monastery",
    "monitor",
    "moped",
    "mortar",
    "mortarboard",
    "mosque",
    "mosquito net",
    "motor scooter",
    "mountain bike",
    "mountain tent",
    "mouse",
    "mousetrap",
    "moving van",
    "muzzle",
    "nail",
    "neck brace",
    "necklace",
    "nipple",
    "notebook",
    "obelisk",
    "oboe",
    "ocarina",
    "odometer",
    "oil filter",
    "organ",
    "oscilloscope",
    "overskirt",
    "oxcart",
    "oxygen mask",
    "packet",
    "paddle",
    "paddlewheel",
    "padlock",
    "paintbrush",
    "pajama",
    "palace",
    "panpipe",
    "paper towel",
    "parachute",
    "parallel bars",
    "park bench",
    "parking meter",
    "passenger car",
    "patio",
    "pay-phone",
    "pedestal",
    "pencil box",
    "pencil sharpener",
    "perfume",
    "Petri dish",
    "photocopier",
    "pick",
    "pickelhaube",
    "picket fence",
    "pickup",
    "pier",
    "piggy bank",
    "pill bottle",
    "pillow",
    "ping-pong ball",
    "pinwheel",
    "pirate",
    "pitcher",
    "plane",
    "planetarium",
    "plastic bag",
    "plate rack",
    "plow",
    "plunger",
    "Polaroid camera",
    "police van",
    "poncho",
    "pool table",
    "pop bottle",
    "pot",
    "potter's wheel",
    "power drill",
    "prayer rug",
    "printer",
    "prison",
    "projectile",
    "projector",
    "puck",
    "punching bag",
    "purse",
    "quill",
    "quilt",
    "racer",
    "racket",
    "radiator",
    "radio",
    "radio telescope",
    "rain barrel",
    "recreational vehicle",
    "reel",
    "reflex camera",
    "refrigerator",
    "remote control",
    "restaurant",
    "revolver",
    "rifle",
    "rocking chair",
    "rotisserie",
    "rubber eraser",
    "rugby ball",
    "rule",
    "running shoe",
    "safe",
    "safety pin",
    "saltshaker",
    "sandal",
    "sarong",
    "sax",
    "scabbard",
    "scale",
    "school bus",
    "schooner",
    "scoreboard",
    "screen",
    "screw",
    "screwdriver",
    "seat belt",
    "sewing machine",
    "shield",
    "shoe shop",
    "shoji",
    "shopping basket",
    "shopping cart",
    "shovel",
    "shower cap",
    "shower curtain",
    "ski",
    "ski mask",
    "sleeping bag",
    "slide rule",
    "sliding door",
    "slot",
    "snorkel",
    "snowmobile",
    "snowplow",
    "soap dispenser",
    "soccer ball",
    "sock",
    "solar dish",
    "sombrero",
    "soup bowl",
    "space bar",
    "space heater",
    "space shuttle",
    "spatula",
    "speedboat",
    "spider web",
    "spindle",
    "sports car",
    "spotlight",
    "stage",
    "steam locomotive",
    "steel arch bridge",
    "steel drum",
    "stethoscope",
    "stole",
    "stone wall",
    "stopwatch",
    "stove",
    "strainer",
    "streetcar",
    "stretcher",
    "studio couch",
    "stupa",
    "submarine",
    "suit",
    "sundial",
    "sunglass",
    "sunglasses",
    "sunscreen",
    "suspension bridge",
    "swab",
    "sweatshirt",
    "swimming trunks",
    "swing",
    "switch",
    "syringe",
    "table lamp",
    "tank",
    "tape player",
    "teapot",
    "teddy",
    "television",
    "tennis ball",
    "thatch",
    "theater curtain",
    "thimble",
    "thresher",
    "throne",
    "tile roof",
    "toaster",
    "tobacco shop",
    "toilet seat",
    "torch",
    "totem pole",
    "tow truck",
    "toyshop",
    "tractor",
    "trailer truck",
    "tray",
    "trench coat",
    "tricycle",
    "trimaran",
    "tripod",
    "triumphal arch",
    "trolleybus",
    "trombone",
    "tub",
    "turnstile",
    "typewriter keyboard",
    "umbrella",
    "unicycle",
    "upright",
    "vacuum",
    "vase",
    "vault",
    "velvet",
    "vending machine",
    "vestment",
    "viaduct",
    "violin",
    "volleyball",
    "waffle iron",
    "wall clock",
    "wallet",
    "wardrobe",
    "warplane",
    "washbasin",
    "washer",
    "water bottle",
    "water jug",
    "water tower",
    "whiskey jug",
    "whistle",
    "wig",
    "window screen",
    "window shade",
    "Windsor tie",
    "wine bottle",
    "wing",
    "wok",
    "wooden spoon",
    "wool",
    "worm fence",
    "wreck",
    "yawl",
    "yurt",
    "web site",
    "comic book",
    "crossword puzzle",
    "street sign",
    "traffic light",
    "book jacket",
    "menu",
    "plate",
    "guacamole",
    "consomme",
    "hot pot",
    "trifle",
    "ice cream",
    "ice lolly",
    "French loaf",
    "bagel",
    "pretzel",
    "cheeseburger",
    "hot dog",
    "mashed potato",
    "head cabbage",
    "broccoli",
    "cauliflower",
    "zucchini",
    "spaghetti squash",
    "acorn squash",
    "butternut squash",
    "cucumber",
    "artichoke",
    "bell pepper",
    "cardoon",
    "mushroom",
    "Granny Smith",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple",
    "banana",
    "jackfruit",
    "custard apple",
    "pomegranate",
    "hay",
    "carbonara",
    "chocolate sauce",
    "dough",
    "meat loaf",
    "pizza",
    "potpie",
    "burrito",
    "red wine",
    "espresso",
    "cup",
    "eggnog",
    "alp",
    "bubble",
    "cliff",
    "coral reef",
    "geyser",
    "lakeside",
    "seashore",
    "valley",
    "volcano",
    "ballplayer",
    "groom",
    "scuba diver",
    "rapeseed",
    "daisy",
    "yellow lady's slipper",
    "corn",
    "acorn",
    "hip",
    "buckeye",
    "horse chestnut",
    "coral fungus",
    "agaric",
    "gyromitra",
    "stinkhorn",
    "earthstar",
    "hen-of-the-woods",
    "bolete",
    "ear",
    "toilet tissue",
]


def get_imagenet_text_prompts(template: str = "a photo of a {}"):
    """Generate text prompts for ImageNet classes."""
    return [template.format(class_name) for class_name in IMAGENET_CLASSES]


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def create_model_from_checkpoint(checkpoint, device: str = "cuda"):
    """Create and load model from checkpoint."""
    config = checkpoint.get("config", {})

    # Extract student model config
    student_cfg = config.get("student", {})

    # Create model
    model = TinySiglipModel(
        vision_model_name=student_cfg.get("vision_model_name", "vit_tiny_patch16_224"),
        vision_dim=student_cfg.get("vision_dim", 384),
        text_vocab_size=student_cfg.get("vocab_size", 32000),
        text_seq_len=config.get("training", {}).get("text_seq_len", 64),
        text_dim=student_cfg.get("text_dim", 384),
        text_layers=student_cfg.get("text_layers", 4),
        text_nhead=student_cfg.get("text_nhead", 8),
        text_ff_dim_multiplier=student_cfg.get("text_ff_dim_multiplier", 4),
        projection_dim=student_cfg.get("projection_dim", 384),
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["student_model"])
    model.eval()

    return model


def load_processor_from_checkpoint(checkpoint_dir: Path):
    """Load processor from checkpoint directory."""
    processor_path = checkpoint_dir / "processor"
    if processor_path.exists():
        try:
            processor = TinySiglipProcessor.from_pretrained(str(processor_path))
            return processor
        except Exception as e:
            print(f"Warning: Could not load processor from checkpoint: {e}")
            return None
    return None


def create_processor_from_config(config, device: str = "cuda"):
    """Create processor from config."""
    student_cfg = config.get("student", {})
    training_cfg = config.get("training", {})

    tokenizer_name = student_cfg.get("tokenizer_name", "google/siglip-base-patch16-224")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
        return None

    from transformers import AutoImageProcessor

    try:
        image_processor = AutoImageProcessor.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Warning: Could not load image processor: {e}")
        image_processor = None

    processor = TinySiglipProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_size=training_cfg.get("image_size", 224),
        max_seq_len=training_cfg.get("text_seq_len", 64),
        use_augmentation=False,  # No augmentation for evaluation
    )

    return processor


def evaluate_imagenet(
    model: TinySiglipModel,
    processor: TinySiglipProcessor,
    imagenet_val_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    logit_scale: float | None = None,
):
    """
    Evaluate model on ImageNet-1k zero-shot classification.

    Args:
        model: TinySiglipModel instance
        processor: TinySiglipProcessor instance
        imagenet_val_path: Path to ImageNet validation set
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        device: Device to run evaluation on
        logit_scale: Optional logit scale (temperature). If None, uses model's default.
    """

    # Create ImageNet dataset with proper preprocessing
    # We'll use a custom dataset wrapper to apply processor's image preprocessing
    class ImageNetDataset(Dataset):
        def __init__(self, imagenet_path, processor):
            self.dataset = ImageNet(root=imagenet_path, split="val", transform=None)
            self.processor = processor

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            # Process image using processor
            processed = self.processor.image_processor(image, return_tensors="pt")
            return processed["pixel_values"].squeeze(0), label

    try:
        imagenet_dataset = ImageNetDataset(imagenet_val_path, processor)
    except Exception as e:
        print(f"Error loading ImageNet dataset: {e}")
        print("Please ensure ImageNet dataset is available at the specified path.")
        return None

    dataloader = DataLoader(
        imagenet_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    # Generate text prompts for all ImageNet classes
    text_prompts = get_imagenet_text_prompts()

    # Tokenize all text prompts
    print("Tokenizing ImageNet class prompts...")
    text_inputs = processor.tokenizer(
        text_prompts,
        padding=True,
        truncation=True,
        max_length=processor.max_seq_len,
        return_tensors="pt",
    )
    text_ids = text_inputs["input_ids"].to(device)  # (1000, seq_len)

    # Compute text features for all classes
    print("Computing text features for all classes...")
    with torch.no_grad():
        # Process text in batches to avoid memory issues
        text_features_list = []
        text_batch_size = 100
        for i in range(0, len(text_ids), text_batch_size):
            batch_text_ids = text_ids[i : i + text_batch_size]
            # Use dummy images (won't affect text features)
            dummy_images = torch.zeros(batch_text_ids.size(0), 3, 224, 224).to(device)
            _, text_features_batch, _, _ = model(dummy_images, batch_text_ids)
            text_features_list.append(text_features_batch)
        text_features_all = torch.cat(text_features_list, dim=0)  # (1000, projection_dim)
        text_features_all = F.normalize(text_features_all, dim=-1)

    # Use logit scale from checkpoint if available, otherwise use default
    if logit_scale is None:
        logit_scale = 1.0

    # Evaluate on validation set
    print("Evaluating on ImageNet validation set...")
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    # Create dummy text_ids for image feature extraction (shape doesn't matter for image features)
    dummy_text_ids = torch.zeros(1, processor.max_seq_len, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # Compute image features (text_ids won't affect image features)
            # Expand dummy_text_ids to match batch size
            batch_dummy_text = dummy_text_ids.expand(batch_size, -1)
            image_features, _, _, _ = model(images, batch_dummy_text)
            image_features = F.normalize(image_features, dim=-1)  # (B, projection_dim)

            # Compute similarity with all class text features
            logits = logit_scale * image_features @ text_features_all.T  # (B, 1000)

            # Get top-1 and top-5 predictions
            _, top5_preds = torch.topk(logits, k=5, dim=1)  # (B, 5)
            top1_preds = top5_preds[:, 0]  # (B,)

            # Check accuracy
            correct_top1 += (top1_preds == labels).sum().item()
            correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

    # Calculate accuracy
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total

    print("\n" + "=" * 60)
    print("ImageNet-1k Zero-Shot Classification Results:")
    print("=" * 60)
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"Total samples: {total}")
    print("=" * 60 + "\n")

    return {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "total_samples": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinySigLIP on ImageNet-1k")
    parser.add_argument(
        "--imagenet-val",
        type=str,
        required=True,
        help="Path to ImageNet validation set",
    )
    parser.add_argument(
        "--resume",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=None,
        help="Logit scale (temperature). If None, uses checkpoint value or default 1.0",
    )

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.resume}...")
    checkpoint = load_checkpoint(args.resume, device=args.device)

    # Get logit scale from checkpoint if available
    logit_scale = args.logit_scale
    if logit_scale is None and "logit_scale" in checkpoint:
        logit_scale_val = checkpoint["logit_scale"]
        logit_scale = logit_scale_val.item() if torch.is_tensor(logit_scale_val) else logit_scale_val
        print(f"Using logit scale from checkpoint: {logit_scale:.4f}")
    elif logit_scale is None:
        logit_scale = 1.0
        print(f"Using default logit scale: {logit_scale}")

    # Create model
    print("Creating model from checkpoint...")
    model = create_model_from_checkpoint(checkpoint, device=args.device)

    # Load or create processor
    checkpoint_dir = Path(args.resume).parent
    processor = load_processor_from_checkpoint(checkpoint_dir)

    if processor is None:
        print("Creating processor from config...")
        config = checkpoint.get("config", {})
        processor = create_processor_from_config(config, device=args.device)

    if processor is None:
        raise ValueError("Could not load or create processor. Please check checkpoint and config.")

    # Evaluate
    results = evaluate_imagenet(
        model=model,
        processor=processor,
        imagenet_val_path=args.imagenet_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        logit_scale=logit_scale,
    )

    if results:
        print("Evaluation completed successfully!")
        print(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
        print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
