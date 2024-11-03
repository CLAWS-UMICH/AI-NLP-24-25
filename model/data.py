# model/data.py

import random
import string
import collections

# =========================
# 1. Define Label Constants
# =========================
WALKING = 0   # Class index 0
RUNNING = 1   # Class index 1
DANCING = 2   # Class index 2
EATING = 3    # Class index 3
SLEEPING = 4  # Class index 4
CODING = 5    # Class index 5

NUM_CLASSES = 6

# List of all possible actions
all_actions = [WALKING, RUNNING, DANCING, EATING, SLEEPING, CODING]

# Action mapping for output (list where index corresponds to class)
action_map_order = [
    "walking",    # 0
    "running",    # 1
    "dancing",    # 2
    "eating",     # 3
    "sleeping",   # 4
    "coding"      # 5
]

# =========================
# 2. Vocabulary and Actions
# =========================

# First, define the basic vocabulary
vocab = {
    "<PAD>": 0,
    "i": 1,
    "am": 2,
    "the": 3,
    "to": 4,
    "in": 5,
    "at": 6,
    "on": 7,
    "with": 8,
    "my": 9,
    "computer": 10,
    "outside": 11,
    "inside": 12,
    "now": 13,
    "fast": 14,
    "slow": 15,
    "and": 16,
    "while": 17,
    "like": 18,
    "love": 19,
    "keep": 20,
    "always": 21,
    "every": 22,
    "day": 23,
    "enjoy": 24,
    "just": 25,
    "still": 26,
    "constantly": 27,
    "currently": 28,
    "started": 29,
    "continue": 30,
    "want": 31,
    "need": 32,
    "time": 33,
    "about": 34,
    "going": 35,
    "over": 36,
    "then": 37,
    "finally": 38,
    "first": 39,
    "prefer": 40,
    "for": 41,
    "have": 42,
    "has": 43,
    "having": 44,
    "take": 45,
    "takes": 46,
    "taking": 47,
    "do": 48,
    "does": 49,
    "doing": 50,
    "go": 51,
    "goes": 52,
    "went": 53,
    "food": 54,
    "meal": 55,
    "move": 56,
    "code": 57,
    "rest": 58,
    "nap": 59,
}

# Action words mapping (base form only)
action_words = {
    WALKING: ["walk"],
    RUNNING: ["run"],
    DANCING: ["dance"],
    EATING: ["eat"],
    SLEEPING: ["sleep"],
    CODING: ["code"]
}

# Verb forms mapping
verb_forms = {
    # Walking
    "walk": WALKING,
    "walks": WALKING,
    "walking": WALKING,
    "to walk": WALKING,
    "went walking": WALKING,
    "go walking": WALKING,
    "goes walking": WALKING,
    "going walking": WALKING,
    "take a walk": WALKING,
    "takes a walk": WALKING,
    "taking a walk": WALKING,
    "went for a walk": WALKING,
    "stroll": WALKING,
    "strolling": WALKING,
    "strolls": WALKING,
    
    # Running
    "run": RUNNING,
    "runs": RUNNING,
    "running": RUNNING,
    "to run": RUNNING,
    "went running": RUNNING,
    "go running": RUNNING,
    "goes running": RUNNING,
    "going running": RUNNING,
    "take a run": RUNNING,
    "takes a run": RUNNING,
    "taking a run": RUNNING,
    "went for a run": RUNNING,
    "jog": RUNNING,
    "jogs": RUNNING,
    "jogging": RUNNING,
    "sprint": RUNNING,
    "sprints": RUNNING,
    "sprinting": RUNNING,
    
    # Dancing
    "dance": DANCING,
    "dances": DANCING,
    "dancing": DANCING,
    "to dance": DANCING,
    "went dancing": DANCING,
    "go dancing": DANCING,
    "goes dancing": DANCING,
    "going dancing": DANCING,
    "do a dance": DANCING,
    "does a dance": DANCING,
    "doing a dance": DANCING,
    "bust a move": DANCING,
    "busts a move": DANCING,
    "busting a move": DANCING,
    "groove": DANCING,
    "grooving": DANCING,
    "grooves": DANCING,
    
    # Eating
    "eat": EATING,
    "eats": EATING,
    "eating": EATING,
    "to eat": EATING,
    "have food": EATING,
    "having food": EATING,
    "has food": EATING,
    "have a meal": EATING,
    "having a meal": EATING,
    "has a meal": EATING,
    "snack": EATING,
    "snacks": EATING,
    "snacking": EATING,
    "munch": EATING,
    "munches": EATING,
    "munching": EATING,
    "dine": EATING,
    "dines": EATING,
    "dining": EATING,
    
    # Sleeping
    "sleep": SLEEPING,
    "sleeps": SLEEPING,
    "sleeping": SLEEPING,
    "to sleep": SLEEPING,
    "take a nap": SLEEPING,
    "takes a nap": SLEEPING,
    "taking a nap": SLEEPING,
    "have a rest": SLEEPING,
    "has a rest": SLEEPING,
    "having a rest": SLEEPING,
    "doze": SLEEPING,
    "dozes": SLEEPING,
    "dozing": SLEEPING,
    "nap": SLEEPING,
    "naps": SLEEPING,
    "napping": SLEEPING,
    "rest": SLEEPING,
    "rests": SLEEPING,
    "resting": SLEEPING,
    
    # Coding
    "code": CODING,
    "codes": CODING,
    "coding": CODING,
    "to code": CODING,
    "program": CODING,
    "programs": CODING,
    "programming": CODING,
    "develop": CODING,
    "develops": CODING,
    "developing": CODING,
    "write code": CODING,
    "writes code": CODING,
    "writing code": CODING,
    "hack": CODING,
    "hacks": CODING,
    "hacking": CODING,
    "debug": CODING,
    "debugs": CODING,
    "debugging": CODING,
}

# Add all action words to vocabulary
next_id = max(vocab.values()) + 1
for words_list in action_words.values():
    for word in words_list:
        if word not in vocab:
            vocab[word] = next_id
            next_id += 1

# =========================
# 3. Sentence Templates
# =========================

# Sentence templates (simplified for better control)
templates = {
    1: [
        "i {0}",
        "i am {0}",
        "i like {0}",
        "i love {0}",
        "i enjoy {0}",
        "i keep {0}",
        "just {0}",
        "still {0}",
        "always {0}",
        "constantly {0}",
        "currently {0}",
        "i started {0}",
        "i continue {0}",
        "i want to {0}",
        "i need to {0}",
        "time to {0}",
        "about to {0}",
        "going to {0}",
    ],
    2: [
        "i {0} and {1}",
        "i like {0} and {1}",
        "i am {0} and {1}",
        "i enjoy {0} and {1}",
        "keep {0} and {1}",
        "love {0} and {1}",
        "always {0} while {1}",
        "i started {0} and {1}",
        "constantly {0} and {1}",
        "i want to {0} and {1}",
        "time for {0} and {1}",
        "{0} while {1}",
        "first {0} then {1}",
        "i prefer {0} over {1}",
    ],
    3: [
        "i {0}, {1}, and {2}",
        "i like {0}, {1}, and {2}",
        "i am {0}, {1}, and {2}",
        "always {0}, {1}, and {2}",
        "love {0}, {1}, and {2}",
        "enjoy {0}, {1}, and {2}",
        "keep {0}, {1}, and {2}",
        "started {0}, {1}, and {2}",
        "want to {0}, {1}, and {2}",
        "time for {0}, {1}, and {2}",
        "first {0}, then {1}, finally {2}",
    ]
}

# =========================
# 4. Utility Functions
# =========================

def tokenize(sentence):
    """Convert a sentence into a list of vocabulary indices"""
    tokens = []
    for word in sentence.lower().split():
        word_clean = word.strip(string.punctuation)
        tokens.append(vocab.get(word_clean, vocab["<PAD>"]))  # Use <PAD> for unknown words
    return tokens

def add_spelling_error(word):
    """Introduce a simple spelling error by swapping two adjacent letters"""
    if len(word) < 2:
        return word
    idx = random.randint(0, len(word) - 2)
    return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]

def detect_actions(sentence):
    """Detect actions in a sentence and return a bitmask label"""
    label = 0
    words = sentence.lower().split()
    
    # Check for all possible phrases (up to 4 words long)
    for i in range(len(words)):
        for j in range(1, 5):  # Check phrases of length 1-4
            if i + j > len(words):
                break
                
            phrase = ' '.join(words[i:i+j])
            if phrase in verb_forms:
                label |= (1 << verb_forms[phrase])
    
    return label

def create_label_from_words(words):
    """Create label from list of action words"""
    label = 0
    for word in words:
        word_clean = word.strip(string.punctuation).lower()
        if word_clean in verb_forms:
            label |= (1 << verb_forms[word_clean])
    return label

def validate_synthetic_data(data):
    """Validate that our synthetic data matches expected labels"""
    print("\nValidating synthetic data...")
    
    for sentence, expected_label in data:
        # Get all the actions that should be in this sentence
        expected_actions = [i for i in range(NUM_CLASSES) if expected_label & (1 << i)]
        
        # Verify each expected action's words appear in the sentence
        for action in expected_actions:
            action_forms = [form for form, a in verb_forms.items() if a == action]
            if not any(form in sentence.lower() for form in action_forms):
                print(f"\nWarning: Expected action {action_map_order[action]} but found no matching words")
                print(f"Sentence: {sentence}")
                print(f"Expected label: {bin(expected_label)}")
                print(f"Action forms: {action_forms}")

def generate_synthetic_data(num_samples=5000):
    """Generate synthetic training data from templates"""
    synthetic_data = []
    
    while len(synthetic_data) < num_samples:
        # 1. Pick random template
        num_actions = random.randint(1, 3)
        template = random.choice(templates[num_actions])
        
        # 2. Pick random actions and get their verb forms
        selected_actions = random.sample(all_actions, num_actions)
        action_forms = []
        combined_label = 0
        
        for action in selected_actions:
            # Get random verb form for this action
            possible_forms = [form for form, a in verb_forms.items() if a == action]
            verb_form = random.choice(possible_forms)
            
            # Maybe add spelling error (10% chance)
            if random.random() < 0.1:
                verb_form = add_spelling_error(verb_form)
                
            action_forms.append(verb_form)
            combined_label |= (1 << action)  # Add to bitmask
            
        # 3. Create sentence from template
        sentence = template.format(*action_forms)
        
        synthetic_data.append((sentence, combined_label))
    
    # Log statistics about the generated data
    print("\nData Generation Statistics:")
    print(f"Total samples generated: {len(synthetic_data)}")
    
    # Count occurrences of each action
    action_counts = collections.defaultdict(int)
    for _, label in synthetic_data:
        for i in range(NUM_CLASSES):
            if label & (1 << i):
                action_counts[action_map_order[i]] += 1
                
    print("\nAction frequencies:")
    for action, count in action_counts.items():
        percentage = (count / num_samples) * 100
        print(f"{action:10}: {count:5d} ({percentage:5.1f}%)")
    
    # Count number of actions per sentence
    action_per_sentence = collections.Counter(bin(label).count('1') for _, label in synthetic_data)
    print("\nActions per sentence distribution:")
    for num_actions, count in sorted(action_per_sentence.items()):
        percentage = (count / num_samples) * 100
        print(f"{num_actions} action(s): {count:5d} ({percentage:5.1f}%)")
    
    # Print some example sentences with their labels
    print("\nExample sentences:")
    examples = {1: [], 2: [], 3: []}
    for sentence, label in synthetic_data:
        num_actions = bin(label).count('1')
        if len(examples[num_actions]) < 2:  # Collect 2 examples for each type
            examples[num_actions].append((sentence, label))
    
    for num_actions, sentence_pairs in examples.items():
        print(f"\n{num_actions} action examples:")
        for sentence, label in sentence_pairs:
            actions = [action_map_order[i] for i in range(NUM_CLASSES) if label & (1 << i)]
            print(f"  - {sentence}")
            print(f"    Label: {bin(label)} ({', '.join(actions)})")
            
    return synthetic_data

# Move vocab generation to after data generation
def build_vocabulary(synthetic_data):
    """Build vocabulary from generated sentences"""
    vocab = {"<PAD>": 0}  # Start with padding token
    word_id = 1
    
    for sentence, _ in synthetic_data:
        for word in sentence.lower().split():
            word = word.strip(string.punctuation)
            if word not in vocab:
                vocab[word] = word_id
                word_id += 1
                
    return vocab

# =========================
# 5. Test Sentences
# =========================
# These are imported by model.py for periodic testing during training
test_sentences = [
    "i am running",
    "i like to code",
    "i am sleeping",
    "i want to dance",
    "i am eating",
    "i am walking outside"
]