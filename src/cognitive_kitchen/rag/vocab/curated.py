"""Tier 2, curated. The ingredient vocabulary for this corpus, decided by hand.

Why hand-written and not model-generated:

    This file is a static artifact. It is built once, committed, and read
    forever after. Nothing about it needs to be reproducible by a cheap model
    at request time, so there is no reason to accept a cheap model's mistakes.
    gpt-4o-mini filed besan as gluten and collapsed coriander seeds into
    coriander leaves; both are wrong, and the second is wrong in a way that
    matters to a cook.

Three consumers depend on this, and only these three:

    graph            node identity -- one ginger node, not sixteen
    hallucination    "ginger" must not read as invented when the context
                     said "1 inch fresh ginger, chopped"
    chat pantry      the user types "ginger"

FAMILIES is the source of truth: canonical name -> every surface form in the
corpus that means it. SPLIT holds lines the PDF wrapped, where one string holds
several ingredients. REJECT holds strings that are not ingredients at all --
method text, dish names, glossary entries, extraction noise.

Anything absent from all three keeps its Tier 1 normalised form as its own
canonical name. That is the safe default: it may be redundant, never wrong.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Allergen categories, keyed by CANONICAL name only.
#
# Deliberate calls:
#   besan, gram flour, rice flour  are gluten-free. besan is chickpea.
#   coconut milk, coconut cream    are not dairy.
#   sesame, poppy, sunflower seed  are seeds, not tree nuts.
#   peanut                         is a legume, but it is a major allergen,
#                                  so it is filed under nut on purpose.
# ---------------------------------------------------------------------------
CATEGORY: dict[str, str] = {
    "ghee": "dairy", "butter": "dairy", "milk": "dairy", "yogurt": "dairy",
    "cream": "dairy", "sour cream": "dairy", "paneer": "dairy",
    "cottage cheese": "dairy", "mozzarella cheese": "dairy",
    "ricotta cheese": "dairy", "mawa": "dairy", "buttermilk": "dairy",
    "khoya": "dairy", "condensed milk": "dairy",

    "wheat flour": "gluten", "all-purpose flour": "gluten", "maida": "gluten",
    "chapatti flour": "gluten", "semolina": "gluten", "vermicelli": "gluten",
    "pizza crust": "gluten", "bread": "gluten", "samosa pastry": "gluten",

    "almond": "nut", "cashew": "nut", "pistachio": "nut", "peanut": "nut",
    "walnut": "nut", "chironji": "nut",

    "chicken": "meat", "lamb": "meat", "mutton": "meat", "goat": "meat",
    "pork": "meat", "beef": "meat",

    "fish": "fish", "prawn": "fish",

    "egg": "egg",
}

# ---------------------------------------------------------------------------
# canonical -> surface forms found in this corpus
# ---------------------------------------------------------------------------
FAMILIES: dict[str, list[str]] = {

    # -- aromatics -------------------------------------------------------
    "onion": ["onions", "onion fine", "med- onions fine", "th onions", "thcup onion",
              "white onions", "onions preferably red onions", "pearl onions/shallots"],
    "onion puree": ["onion pureed", "onions pureed"],
    "spring onion": ["spring onions", "i spring onion including green leaves"],
    "garlic": ["garlic fine", "garlic ground", "garlic halves"],
    "garlic paste": [],
    "ginger": ["fresh ginger", "fresh ginger root", "f fresh ginger", "little ginger",
               "ginger fine", "ginger into strips", "powdered ginger"],
    "ginger paste": [],
    "ginger garlic paste": ["garlic paste powder", "ginger paste powder"],

    # -- chillies: form changes the ingredient, so these stay apart -------
    "green chilli": ["green chillies", "fresh green chilli", "fresh green chillies",
                     "fresh hot green chilli", "hot green chillies", "green chills",
                     "green chillies slit", "green chillies slit into two",
                     "hot green chills slit into", "i fresh green chilli",
                     "seedless green chilli fresh", "fresh chillies",
                     "fresh mild chillies seeds removed", "jalapeno serrano pepper"],
    "dried red chilli": ["red chilli", "red chillies", "dried red chillies",
                         "dried red chilli", "dried red chilli pods", "dry red chillies",
                         "red dry chillies", "hot red dried chilli", "red chile",
                         "red chilli into two", "red chillies up to",
                         "byadagi red chilli", "byadgi red chilli"],
    "chilli powder": ["red chilli powder", "chile powder", "tspn red chilli powder",
                      "cayenne", "cayenne pepper", "paprika", "dried red pepper flakes"],

    # -- coriander: three distinct ingredients, never merged ---------------
    "coriander leaves": ["cilantro", "cilantro leaves", "cilantro/coriander leaves",
                         "coriander / cilantro leaves", "fresh coriander",
                         "fresh coriander leaves", "fresh cilantro",
                         "fresh cilantro/coriander leaves", "cilantro fresh",
                         "cilantro coriander leaves", "fresh cilantro coriander",
                         "very fresh green coriander", "coriander",
                         "coriander leaves to garish", "cilantro stems root removed"],
    "coriander seeds": ["tspn coriander seeds", "coriander seeds ground"],
    "coriander powder": ["ground coriander", "ground coriander seeds", "i ground coriander",
                         "tspn dhania powder", "coriander paste powder",
                         "coriander seeds roasted & powdered"],

    # -- cumin: whole seed and ground are used differently ----------------
    "cumin seeds": ["cumin", "cumin seed", "l cumin seeds", "tspn cumin", "tspn jeera",
                    "roasted cumin seeds", "black cumin seeds", "caraway seeds"],
    "cumin powder": ["ground cumin", "roasted jeera powder", "ground roasted cumin seeds",
                     "cumin seeds roasted ground", "cumin seeds roasted into powder"],

    # -- turmeric: this book only ever uses the powder, so merging is safe -
    "turmeric": ["turmeric powder", "ground turmeric", "tspn turmeric"],

    # -- other spices -----------------------------------------------------
    "mustard seeds": ["mustard", "black mustard seed", "black mustard seeds",
                      "brown mustard seeds", "tspn mustard", "tspn mustard seeds",
                      "tspn black mustard seeds"],
    "mustard powder": [],
    "asafoetida": ["hing", "asafoetida powder", "good asafoetida", "hing / asafetida"],
    "cardamom": ["cardamoms", "cardamom pods", "cardamom pods cracked", "cardamom seeds",
                 "to cardamom pods", "green cardamom pods ground"],
    "cardamom powder": ["ground cardamom", "cardamom pods ground", "th cardamom powder",
                        "powdered cardamom"],
    "cinnamon": ["ground cinnamon"],
    "black pepper": ["black peppercorns", "peppercorns", "pepper corns", "ground pepper",
                     "ground black pepper", "white pepper", "red pepper", "green pepper"],
    "fenugreek seeds": ["methi", "methi seeds", "tspn fenugreek"],
    "fenugreek leaves": ["methi leaves"],
    "fennel seeds": ["ground fennel"],
    "nutmeg": ["ground nutmeg"],
    "saffron": ["powdered saffron"],
    "garam masala": ["garam masala powder", "tspn garam masala", "garam masala eatable",
                     "mixed spices"],
    "curry powder": [],
    "sambar powder": ["heaped tspn sambar powder", "tspn sambar powder"],
    "chaat masala": [],
    "amchur": ["mango powder called amchur", "amchoor powder"],
    "poppy seeds": ["white poppy seeds", "khas-khas",
                    "white poppy seeds dry roasted pan"],
    "sesame seeds": ["til"],
    "kalpasi": ["kalpasi / black stone flower"],
    "bay leaf": ["bay leaves"],
    "curry leaves": ["fresh curry leaves", "dozen curry leaves"],
    "mint leaves": ["mint", "fresh mint", "dry mint"],
    "edible camphor": ["camphor"],
    "silver leaf": [],
    "ratan jot": [],

    # -- dairy -------------------------------------------------------------
    "ghee": ["clarified butter"],
    "butter": ["cup butter", "salt free butter"],
    "milk": ["full fat milk", "cold milk preferably", "fresh chenna cows milk"],
    "yogurt": ["yoghurt", "plain yogurt", "yogurt/curds", "sour curd", "curd"],
    "buttermilk": [],
    "cream": ["heavy cream", "yoghurt cream"],
    "sour cream": [],
    "paneer": ["pannir"],
    "cottage cheese": [],
    "mozzarella cheese": [],
    "ricotta cheese": [],
    "mawa": ["khoa"],

    # -- flours and grains -------------------------------------------------
    "besan": ["besan flour", "besan -chickpea flour", "gram flour",
              "chickpea flour", "besan gram flour"],
    "rice flour": ["rice powder", "measures rice flour"],
    "urad flour": [],
    "wheat flour": [],
    "all-purpose flour": ["flour", "maida", "one maida", "rice all purpose flour"],
    "chapatti flour": [],
    "semolina": ["rava", "one semolina/rava", "sooji"],
    "vermicelli": ["semiya"],
    "arrowroot": ["arrarot"],
    "rice": ["long grain rice", "white rice", "uncooked rice", "basmati rice",
             "basmati other long grain rice", "long grain rice preferably basmati",
             "sona masoori rice basmati rice", "h long grain rice"],
    "poha": ["pohe", "flattened rice", "poha / aval / pressed rice"],
    "puffed rice": [],
    "sago": ["sago tapioca"],

    # -- pulses ------------------------------------------------------------
    "urad dal": ["urad daal", "urad dhal", "tspn urad dal", "urad dal dry",
                 "tspn white urad dal", "black gram", "black gram dal",
                 "black gram dal picked over"],
    "toor dal": ["toor daal", "toovar dal", "arhar", "toor dal dry",
                 "arhar/toor daal", "daal", "dhal", "dal variety"],
    "chana dal": ["chana daal", "tspn chana dal", "channa dal dry", "chana daal seeds"],
    "moong dal": ["moong daal", "mung dal", "green moong dhal",
                  "sprouted green moong dal", "green gram",
                  "sprouted green gram moong dal"],
    "masoor dal": ["masoor dhal", "masur daal", "red lentils"],
    "chickpeas": ["chick peas", "canned chickpeas", "garbanzo", "chana"],
    "yellow split peas": [],
    "red kidney bean": ["rajma"],
    "roasted gram": ["pottukadalai", "daliya"],

    # -- vegetables --------------------------------------------------------
    "potato": ["potatoes", "baby potatoes", "baking potato", "round potatoes",
               "oiled potatoes", "to potatoes",
               "potatoes their jackets allowed to cool"],
    "tomato": ["tomatoes", "fresh tomatoes", "tomato fine", "tomatoes into",
               "to tomatoes", "plum tomatoes"],
    "tomato puree": ["tomato pureed", "tomatoes pureed"],
    "tomato paste": [],
    "tomato sauce": [],
    "carrot": ["carrots"],
    "cucumber": ["cucumbers", "green cucumbers", "mangalore cucumber"],
    "cauliflower": ["cauliflower flowerets into cubes"],
    "cabbage": ["head cabbage fine"],
    "brinjal": ["eggplant", "egg plant", "round brinjals", "sized eggplant",
                "one bhima eggplant"],
    "capsicum": ["bell peppers", "green peppers"],
    "green peas": ["peas", "shelled peas", "shelled green peas", "fresh green peas",
                   "dried green peas", "frozen peas",
                   "fresh green peas / pacha pattani"],
    "green beans": ["fresh green beans"],
    "okra": ["okra lengthwise into"],
    "spinach": ["spinach leaves"],
    "broccoli": ["broccoli florets"],
    "zucchini": [],
    "bitter gourd": ["med bitter gourds"],
    "ivy gourd": [],
    "lettuce": ["lettuce leaves fine"],
    "mixed vegetables": ["mixed", "vegetable", "vegetables mixed",
                         "random vegetables"],

    # -- fruit -------------------------------------------------------------
    "lemon": ["lemons"],
    "lemon juice": ["fresh lemon juice", "juice lemon", "lemon/lime juice",
                    "lime/lemon juice"],
    "lime": [],
    "lime juice": ["juice one lime"],
    "mango": ["cup mango"],
    "mango pulp": ["mango pulp from flesh mangoes"],
    "banana": ["bananas"],
    "raisins": ["seedless raisins", "sultanas", "kishmish", "raisins some"],
    "dates": ["dry dates"],
    "dry fruits": [],
    "tamarind": ["tamarind - gooseberry size", "tamarind about size lemon",
                 "lemon size tamarind about water"],
    "tamarind paste": ["tamarind concentrate", "tspn tamarind paste", "tamcon paste"],
    "tamarind pulp": ["tamarind pulp instant tamarind", "onions tamarind juice"],

    # -- nuts --------------------------------------------------------------
    "almond": ["almonds", "blanched almonds", "ground almonds",
               "slivered blanched almonds", "blanch almonds remove skin"],
    "cashew": ["cashews", "cashew nuts", "kaju", "fried cashew nuts",
               "cashews bits halves", "cashews some", "cashew nut"],
    "pistachio": ["pistachios", "pista", "blanched pistachio", "blanched pistachios",
                  "unsalted pistachios"],
    "peanut": ["peanuts", "groundnuts", "peanut bits", "roasted peanuts",
               "salted peanuts", "unsalted peanuts"],
    "chironji": ["chironji seeds", "chirongi"],
    "sunflower seeds": [],

    # -- coconut -----------------------------------------------------------
    "coconut": ["fresh coconut", "scrapped coconut", "desiccated coconut",
                "dry coconut", "coconut flakes"],
    "coconut milk": [],

    # -- meat, fish, egg ---------------------------------------------------
    "chicken": ["chicken breast", "chicken breast halves", "chicken drumsticks",
                "roasting chicken", "roasted chicken breast", "ingredients chicken",
                "chicken drumsticks/ thighs/ breast", "chicken skin removed into",
                "roasting chicken into bite size"],
    "lamb": ["g lamb", "lean lamb into", "leg lamb boned", "lamb goat shoulder"],
    "mutton": [],
    "fish": ["fish into"],
    "prawn": ["prawns", "shrimps de-veined", "shrimp"],
    "egg": ["eggs", "hard eggs"],

    # -- fats and liquids --------------------------------------------------
    "oil": ["cooking oil", "vegetable oil", "refined oil", "tbspn oil", "tbspoon oil",
            "oil - use wok", "groundnut oil"],
    "sesame oil": ["light sesame oil", "tbspn oil preferably sesame"],
    "water": ["hot water", "boiling water", "water at room temperature",
              "water if you like", "up to water"],
    "vinegar": ["white vinegar"],
    "rose water": ["tbs rose water", "rose water drops"],
    "vanilla": [],
    "coffee": ["instant coffee powder"],
    "tea": ["loose black tea"],

    # -- sweeteners and leaveners -----------------------------------------
    "sugar": ["granulated sugar", "powdered sugar", "sugar cup", "to sugar",
              "brown sugar", "s sugar"],
    "sugar syrup": ["syrup"],
    "jaggery": ["tbspn jaggery"],
    "honey": [],
    "baking soda": ["sodium bicarbonate"],
    "baking powder": ["double acting baking powder"],

    # -- salt --------------------------------------------------------------
    "salt": ["s salt", "thteaspoon salt"],

    # -- prepared items ----------------------------------------------------
    "mild curry paste": [],
    "sweet chutney": ["sweet mango chutney"],
    "potato chips": [],
    "pizza crust": ["pre baked pizza crust"],
    "samosa pastry": ["samosa sheets / samosa patti", "pastry"],
    "food colour": ["orange food color"],
    "ice cubes": ["ice-cubes"],
}

# ---------------------------------------------------------------------------
# Lines the PDF wrapped: one string, several ingredients.
# ---------------------------------------------------------------------------
SPLIT: dict[str, list[str]] = {
    "almonds cashews fried": ["almond", "cashew"],
    "amchoor powder/mango powder": ["amchur"],
    "apple pear nectarine plums guava mango": ["apple", "pear", "nectarine", "plum",
                                               "guava", "mango"],
    "baking powder cardamom water making dough": ["baking powder", "cardamom", "water"],
    "bananas lemon juice": ["banana", "lemon juice"],
    "bisquick sour cream sugar water": ["sour cream", "sugar", "water"],
    "black dal picked over": ["urad dal"],
    "black peppercorns cinnamon": ["black pepper", "cinnamon"],
    "boiling water yogurt salt": ["water", "yogurt", "salt"],
    "butter vegetable oil": ["butter", "oil"],
    "capsicum eggplant choko cucumber etc": ["capsicum", "brinjal", "cucumber"],
    "cardamom powder nuts raisins": ["cardamom powder", "raisins"],
    "cardamom powder vanilla": ["cardamom powder", "vanilla"],
    "carrots saut ed ghee": ["carrot", "ghee"],
    "cashew nut pistachio oil": ["cashew", "pistachio", "oil"],
    "cauliflower green beans etc": ["cauliflower", "green beans"],
    "chana daal ginger": ["chana dal", "ginger"],
    "chilli powder yogurt": ["chilli powder", "yogurt"],
    "chilly powder ginger": ["chilli powder", "ginger"],
    "chironji seeds pistachios": ["chironji", "pistachio"],
    "cinnamon tspn turmeric": ["cinnamon", "turmeric"],
    "clarified butter sugar": ["ghee", "sugar"],
    "coconut coriander leaves garnishing": ["coconut", "coriander leaves"],
    "coconut desiccated coconut water": ["coconut", "water"],
    "cooking oil ghee": ["oil", "ghee"],
    "coriander fresh mint": ["coriander leaves", "mint leaves"],
    "coriander powder potatoes": ["coriander powder", "potato"],
    "cumin coriander turmeric chilli powders": ["cumin powder", "coriander powder",
                                               "turmeric", "chilli powder"],
    "cumin mustard seeds": ["cumin seeds", "mustard seeds"],
    "cumin seeds peppercorns roasted & powdered": ["cumin seeds", "black pepper"],
    "cups shelled green peas carrots": ["green peas", "carrot"],
    "curry leaves salt": ["curry leaves", "salt"],
    "curry powder curd": ["curry powder", "yogurt"],
    "dash lemon salt": ["lemon juice", "salt"],
    "decorating almond colour cardamom cashew nuts": ["almond", "food colour",
                                                      "cardamom", "cashew"],
    "fresh coconut coconut flakes": ["coconut"],
    "fresh coriander mint": ["coriander leaves", "mint leaves"],
    "fresh coriander mint leaves": ["coriander leaves", "mint leaves"],
    "fresh curry leaves dried curry leaves": ["curry leaves"],
    "fresh fenugreek/methi leaves dried": ["fenugreek leaves"],
    "fresh red green chillies": ["green chilli", "dried red chilli"],
    "garbanzo beans tin": ["chickpeas"],
    "garlic green chillies": ["garlic", "green chilli"],
    "garnish tomato wedges": ["tomato"],
    "ghee / butter": ["ghee", "butter"],
    "ghee brushing bread": ["ghee"],
    "ghee butter": ["ghee", "butter"],
    "ghee oil": ["ghee", "oil"],
    "ghee oil eggs": ["ghee", "oil", "egg"],
    "ghee oil shallow frying": ["ghee", "oil"],
    "ghee vegetable oil": ["ghee", "oil"],
    "ginger & garlic": ["ginger", "garlic"],
    "ginger salt": ["ginger", "salt"],
    "green cardamom kishmish": ["cardamom", "raisins"],
    "green chillies red chilli": ["green chilli", "dried red chilli"],
    "green/red chillies": ["green chilli", "dried red chilli"],
    "ground cardamom cinnamon": ["cardamom powder", "cinnamon"],
    "honey & ginger marinade": ["honey", "ginger"],
    "honey sugar": ["honey", "sugar"],
    "hot stock water": ["water"],
    "jaggery brown sugar": ["jaggery", "sugar"],
    "lime juice salt": ["lime juice", "salt"],
    "little jaggery sugar": ["jaggery", "sugar"],
    "little oil roasting seasoning salt": ["oil", "salt"],
    "med tomatoes cardamoms": ["tomato", "cardamom"],
    "milk kesar rice": ["milk", "saffron", "rice"],
    "milk yogurt": ["milk", "yogurt"],
    "mustard chana dal curry leaves seasoning": ["mustard seeds", "chana dal",
                                                 "curry leaves"],
    "mustard cumin seeds": ["mustard seeds", "cumin seeds"],
    "mustard seeds curry leaves oil seasoning": ["mustard seeds", "curry leaves", "oil"],
    "mutton tomatoes coriander fat": ["mutton", "tomato", "coriander leaves"],
    "onions coriander": ["onion", "coriander leaves"],
    "onions curry leaves": ["onion", "curry leaves"],
    "onions ratan jot": ["onion", "ratan jot"],
    "papaya ginger": ["papaya", "ginger"],
    "peaches syrup": ["peach", "sugar syrup"],
    "peanuts/cashew nuts split halves": ["peanut", "cashew"],
    "pista & almond": ["pistachio", "almond"],
    "pistachio milk": ["pistachio", "milk"],
    "pistachio nuts cardamom qty": ["pistachio", "cardamom"],
    "pitted cherries pineapple kiwi seedless grapes": ["cherry", "pineapple", "kiwi",
                                                       "grapes"],
    "plum tomatoes water": ["tomato", "water"],
    "potatoes coriander powder": ["potato", "coriander powder"],
    "potatoes onions": ["potato", "onion"],
    "raisins almonds pistachio like": ["raisins", "almond", "pistachio"],
    "random vegetables butter": ["mixed vegetables", "butter"],
    "rice mung dal": ["rice", "moong dal"],
    "rice rice powder coconut salt": ["rice", "rice flour", "coconut", "salt"],
    "roast methi chana daal asafoetida": ["fenugreek seeds", "chana dal", "asafoetida"],
    "roasted chick peas ground besan": ["chickpeas", "besan"],
    "rose water cashew nut": ["rose water", "cashew"],
    "rose water ghee": ["rose water", "ghee"],
    "saffron/turmeric": ["saffron", "turmeric"],
    "salt &amp water": ["salt", "water"],
    "salt black pepper": ["salt", "black pepper"],
    "salt chilli powder": ["salt", "chilli powder"],
    "salt cilantro/coriander leaves": ["salt", "coriander leaves"],
    "salt free butter ghee": ["butter", "ghee"],
    "salt good curry leaves": ["salt", "curry leaves"],
    "salt gr chillies": ["salt", "green chilli"],
    "salt ground pepper": ["salt", "black pepper"],
    "salt lime juice": ["salt", "lime juice"],
    "salt pepper": ["salt", "black pepper"],
    "semolina sugar fat cardamom - water": ["semolina", "sugar", "cardamom", "water"],
    "sesame seeds lemon juice": ["sesame seeds", "lemon juice"],
    "spinach ghee oil": ["spinach", "ghee", "oil"],
    "sugar ice": ["sugar", "ice cubes"],
    "sugar sugar": ["sugar"],
    "sugar syrup": ["sugar syrup"],
    "sultanas raisins": ["raisins"],
    "sugar chirongi": ["sugar", "chironji"],
    "tbspn ghee/melted butter turned brown": ["ghee", "butter"],
    "tbspn sesame seeds roasted dry": ["sesame seeds"],
    "to ghee melted butter": ["ghee", "butter"],
    "to sugar combination sugar honey": ["sugar", "honey"],
    "tomato cucumber": ["tomato", "cucumber"],
    "tomatoes juice": ["tomato"],
    "tspn cumin turmeric": ["cumin seeds", "turmeric"],
    "turmeric salt": ["turmeric", "salt"],
    "urad dal turmeric": ["urad dal", "turmeric"],
    "water salt": ["water", "salt"],
    "yellow split peas chilli powder turmeric": ["yellow split peas", "chilli powder",
                                                 "turmeric"],
    "yellow split peas red lentils": ["yellow split peas", "masoor dal"],
    "yogurt buttermilk": ["yogurt", "buttermilk"],
    "to cabbage leaves": ["cabbage"],
    "to fresh peaches": ["peach"],
    "green cardamom pods ground": ["cardamom powder"],
    "powdered cardamom cinnamon nutmeg": ["cardamom powder", "cinnamon", "nutmeg"],
    "cardamom fat": ["cardamom"],
    "maida fat": ["all-purpose flour"],
    "dry masala ingredients": ["garam masala"],
    "leftover rice poha": ["rice", "poha"],
    "idli rice dosa rice": ["rice"],
    "measure urad flour": ["urad flour"],
    "ingredients rice flour": ["rice flour"],
    "dry coconut little water": ["coconut", "water"],
    "long grain rice ghee": ["rice", "ghee"],
    "oil making dosas": ["oil"],
    "oil to make dosas": ["oil"],
    "tbspn ghee/ butter turned brown": ["ghee", "butter"],
    "to ghee butter": ["ghee", "butter"],
}

# ---------------------------------------------------------------------------
# Not ingredients: method text, dish names, glossary entries, extraction noise.
# ---------------------------------------------------------------------------
REJECT: set[str] = {
    "add flavor to curries rice", "add milk stir continuously",
    "another term green vegetables", "as much fat as possible", "availability",
    "available at any indian grocery store", "banana oranges are must",
    "bhujia /sev / omapodi", "cabbage size", "called differently", "chumchum chuka",
    "coconut fried dal chutney pottukadalai chutney", "colander", "corn recipe",
    "cover dough damp cloth", "dal", "de- cm cubes", "decoration given to food item",
    "deep fry", "description", "dhokla", "dhum", "directions fried bread puffs recipe",
    "directions naan recipe", "dissolve saffron milk boil milk handy", "dosa", "drops",
    "fat", "filling", "filling mawa", "flat metal plate similar to skillet", "garnish",
    "grating garnish halwa idli", "green rice upma", "green gram rice upma",
    "green sundal sweet n spice versions", "griddle gravy", "grind to paste", "ground s",
    "gtm", "gulab jamun", "heat fat fry rice", "help vegetable grater", "hours",
    "ingredients", "into", "into cubes", "ivy gourd jaggery jilebi", "jhangri kaja flat",
    "kaja round", "kesari khara kheer khoa", "koottu kurma khofta",
    "korma cooking sauce jar", "laddu lassi", "make soft clough without reading",
    "make stiff but pliable dough", "masala", "masala ingredients", "masala preparation",
    "method", "method qty ingredients", "method to roll out dough",
    "mild dal variety choice vegetables", "mix together", "mixed fruit season",
    "mixed vets", "more kozhambhu powder", "namkeen nan", "pakora/pakoda", "pan",
    "papad", "paratha", "payasam pulao", "preparation", "pulusu", "puri", "qty",
    "qty ingredients", "random spices", "rasam", "rasgolla rawa", "readymade puri",
    "riata", "roasted dal / daliya / pottukadalai", "roasted ground cumin as garnish",
    "salad dressing", "sambar samosa", "seasoning", "serving", "set aside minutes",
    "seven burfi", "seven burfi coconut burfi halkova",
    "shrikand semiya sandesh sesame sev", "spices",
    "spicy curry made gravy variety vegetables", "stuffing", "subzi", "tamcom",
    "tamcon", "tamcon flat", "up to water", "using little ones bag about five", "veg",
    "vegetables", "vegetables etc", "view complete seven burfi recipe",
    "wash soak rice four hours", "wooden skewers", "one to eggplant", "th onions",
    "as garnish", "boiling water",
}


def build() -> dict[str, dict[str, object]]:
    """Invert FAMILIES/SPLIT/REJECT into the flat map the consumers read."""
    alias: dict[str, str] = {}
    for canonical, forms in FAMILIES.items():
        alias[canonical] = canonical
        for form in forms:
            alias[form] = canonical

    out: dict[str, dict[str, object]] = {}
    for name in REJECT:
        out[name] = {"canonical": None, "category": "none", "split": None}
    for name, parts in SPLIT.items():
        out[name] = {"canonical": None, "category": "none",
                     "split": [alias.get(p, p) for p in parts]}
    for name, canonical in alias.items():
        out[name] = {"canonical": canonical,
                     "category": CATEGORY.get(canonical, "none"), "split": None}
    return out