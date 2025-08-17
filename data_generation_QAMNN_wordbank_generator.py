import json
import random
import itertools

def generate_dataset():
    """Generates a large, random dataset for the QAM."""
    print("\n   - You have chosen to generate a synthetic dataset for a Quantum Associative Memory Neural Network (QAMNN) suitable for creative.")

# --- Word Bank ---
# A large bank of words from various domains to ensure variety.
    WORD_BANK = [
    # Original 100
    "Art", "Science", "Nature", "Technology", "Emotion", "Time", "Space", "Order", "Chaos", "Dream", "Logic", "Ethics", "Truth",
    "Reality", "Consciousness", "Identity", "Purpose", "Morality", "Virtue", "Freedom", "Matter", "Energy", "Gravity", "Quantum", 
    "Relativity", "Universe", "Galaxy", "Planet", "Star", "Atom", "Life", "Death", "Evolution", "Genetics", "Cell", "Organism", "Ecosystem",
    "Virus", "Brain", "Instinct", "Data", "Algorithm", "Network", "AI", "Simulation", "Code", "Hardware", "Software", "Encryption",
    "Interface", "Music", "Literature", "Poetry", "Dance", "Sculpture", "Architecture", "Design", "Narrative", "Symbol", "Metaphor",
    "Society", "Culture", "History", "Language", "Myth", "Ritual", "Family", "Community", "Power", "Justice", "Economy", "Market",
    "Trade", "Labor", "Capital", "Government", "Law", "War", "Peace", "Diplomacy", "Fear", "Joy", "Sadness", "Anger", "Love", "Hate", "Empathy",
    "Memory", "Perception", "Motivation", "Water", "Fire", "Earth", "Air", "Metal", "Wood", "Crystal", "Light", "Shadow", "Sound",
    "Velocity", "Acceleration", "Mass", "Force", "Wave", "Particle", "Spacetime", "Singularity", "Cosmology", "Astrophysics",
    "Biology", "Chemistry", "Physics", "Geology", "Ecology", "Botany", "Zoology", "Mycology", "Paleontology", "Meteorology",
    "DNA", "RNA", "Protein", "Enzyme", "Molecule", "Compound", "Reaction", "Bond", "Valence", "Isotope",
    "Mind", "Thought", "Subconscious", "Psyche", "Behavior", "Cognition", "Neuroscience", "Psychology", "Sociology", "Anthropology",
    "Philosophy", "Theology", "Ideology", "Aesthetics", "Epistemology", "Metaphysics", "Ontology", "Phenomenology", "Existentialism", "Humanism",
    "Painting", "Drawing", "Photography", "Cinema", "Theater", "Opera", "Symphony", "Sonata", "Rhythm", "Harmony",
    "Prose", "Verse", "Stanza", "Character", "Plot", "Theme", "Genre", "Trope", "Allusion", "Irony",
    "State", "Nation", "Empire", "Republic", "Democracy", "Monarchy", "Oligarchy", "Anarchy", "Utopia", "Dystopia",
    "Revolution", "Rebellion", "Protest", "Movement", "Activism", "Propaganda", "Ideology", "Hegemony", "Sovereignty", "Geopolitics",
    "Currency", "Debt", "Credit", "Asset", "Liability", "Equity", "Stock", "Bond", "Derivative", "Commodity",
    "Industry", "Agriculture", "Manufacturing", "Service", "Automation", "Robotics", "Globalization", "Logistics", "Infrastructure", "Urbanization",
    "Circuit", "Processor", "Memory", "Storage", "Database", "Cloud", "Internet", "Cybersecurity", "Blockchain", "Cryptocurrency",
    "VirtualReality", "AugmentedReality", "MachineLearning", "DeepLearning", "NeuralNetwork", "ExpertSystem", "AGI", "Transhumanism", "Bioengineering", "Nanotechnology",
    "Belief", "Faith", "Dogma", "Doctrine", "Spirituality", "Divinity", "Sacred", "Profane", "Enlightenment", "Nirvana",
    "Health", "Medicine", "Disease", "Pandemic", "Vaccine", "Surgery", "Pharmacy", "Therapy", "Wellness", "Nutrition",
    "Color", "Shape", "Form", "Texture", "Pattern", "Symmetry", "Asymmetry", "Composition", "Perspective", "Scale",
    "Heat", "Cold", "Pressure", "Density", "Viscosity", "Elasticity", "Plasticity", "Magnetism", "Electricity", "Radiation",
    "Mountain", "River", "Ocean", "Desert", "Forest", "Jungle", "Tundra", "Swamp", "Volcano", "Glacier",
    "Sun", "Moon", "Eclipse", "Constellation", "Nebula", "Supernova", "BlackHole", "Quasar", "Pulsar", "Exoplanet",
    "King", "Queen", "Prince", "Princess", "Noble", "Peasant", "Knight", "Priest", "Shaman", "Oracle",
    "Sword", "Shield", "Arrow", "Armor", "Castle", "Siege", "Tactic", "Strategy", "Victory", "Defeat",
    "Happiness", "Suffering", "Desire", "Aversion", "Attachment", "Detachment", "Compassion", "Forgiveness", "Gratitude", "Resilience",
    "Childhood", "Adolescence", "Adulthood", "OldAge", "Birth", "Growth", "Maturity", "Decline", "Legacy", "Ancestor",
    "Communication", "Information", "Knowledge", "Wisdom", "Understanding", "Ignorance", "Deception", "Misinformation", "Propaganda", "Rhetoric",
    "Food", "Water", "Shelter", "Clothing", "Tool", "Weapon", "Fire", "Wheel", "Sail", "Engine",
    "Artisan", "Merchant", "Farmer", "Soldier", "Scholar", "Scribe", "Builder", "Healer", "Leader", "Follower",
    "Past", "Present", "Future", "Eternity", "Moment", "Duration", "Cycle", "Linearity", "Paradox", "Causality",
    "Possibility", "Probability", "Certainty", "Randomness", "Determinism", "FreeWill", "Fate", "Destiny", "Chance", "Luck",
    "Good", "Evil", "Neutral", "Alignment", "Balance", "Duality", "Trinity", "Unity", "Singularity", "Plurality",
    "Creation", "Destruction", "Preservation", "Transformation", "Stasis", "Flux", "Flow", "Equilibrium", "Homeostasis", "Apoptosis",
    "Abstraction", "Concretion", "Instance", "Class", "Object", "Attribute", "Method", "Function", "Variable", "Constant",
    "Vector", "Matrix", "Tensor", "Scalar", "Field", "Graph", "Tree", "Set", "Sequence", "Map",
    "Analysis", "Synthesis", "Hypothesis", "Theory", "Law", "Model", "Framework", "Paradigm", "Axiom", "Theorem",
    "Induction", "Deduction", "Abduction", "Inference", "Argument", "Premise", "Conclusion", "Fallacy", "Bias", "Heuristic",
    "Language", "Grammar", "Syntax", "Semantics", "Pragmatics", "Phonology", "Morphology", "Etymology", "Lexicon", "Dialect",
    "Civilization", "Tribe", "Clan", "Nomad", "Settler", "City", "Metropolis", "Megalopolis", "Suburb", "Rural",
    ]

    # Generate Concepts
    # --------------------
    while True:
        num_concepts = int(input("\nHow many concepts would you like to generate : "))
        if num_concepts < len(WORD_BANK):
            concepts = sorted(random.sample(WORD_BANK, num_concepts))
            print(f"   - Generated {len(concepts)} unique concepts.")
            break
        else:
            print(f"   - Error: Requested {num_concepts} concepts, but the word bank only has {len(WORD_BANK)}.")

    # Generate Associations
    # ------------------------
    possible_pairs = list(itertools.combinations(concepts, 2))

    while True:
        num_associations = int(input("\nHow many associations would you like to generate? : "))
        if num_associations <= len(concepts) * (len(concepts) - 1) // 2:
            num_to_generate = num_associations
            # Sample the desired number of pairs
            associations = random.sample(possible_pairs, num_to_generate)
            # Convert tuples to lists for JSON standard
            associations = [list(pair) for pair in associations]
            print(f"   - Generated {len(associations)} unique associations.")
            break
        else:
            print(f"   - Error: Requested {num_associations} associations, but only {len(concepts) * (len(concepts) - 1) // 2} are possible with {len(concepts)} concepts.")

    # Generate Themes
    # ------------------
    num_themes = int(input("\nHow many themes would you like to generate : "))
    themes = {}
    theme_prefixes = ["The Dynamics of", "A Study in", "The Paradox of", "Foundations of", "The Art of", "Systems of", "The Nature of", "Rethinking", "The Future of", "An Inquiry into"]
    theme_suffixes = ["Systems", "Dynamics", "Paradigms", "Structures", "Networks", "Ethics", "Aesthetics", "Futures", "Realities", "Frameworks"]

    for _ in range(num_themes):
        # Generate a unique theme name
        while True:
            prefix = random.choice(theme_prefixes)
            suffix_concept = random.choice(concepts)
            theme_name = f"{prefix} {suffix_concept}"
            if theme_name not in themes:
                break
        
        # Associate 2 to 5 concepts with the theme
        num_related_concepts = random.randint(2, 5)
        related_concepts = random.sample(concepts, num_related_concepts)
        themes[theme_name] = related_concepts
    print(f"   - Generated {len(themes)} unique themes.")


    # Write to Files
    # -----------------
    print("\n4. Writing data to JSON files...")
    try:
        with open("JSON-files/data_generation_QAMNN_concepts.json", "w") as f:
            json.dump(concepts, f, indent=4)
        print("   - Successfully wrote data_generation_QAMNN_concepts.json")

        with open("JSON-files/data_generation_QAMNN_associations.json", "w") as f:
            json.dump(associations, f, indent=4)
        print("   - Successfully wrote data_generation_QAMNN_associations.json")

        with open("JSON-files/data_generation_QAMNN_themes.json", "w") as f:
            json.dump(themes, f, indent=4)
        print("   - Successfully wrote data_generation_QAMNN_themes.json")
    except IOError as e:
        print(f"   - Error writing files: {e}")

    print("\n   - Done.")
