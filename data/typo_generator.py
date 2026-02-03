import random

def add_typo(word):

    if len(word) < 2:
        return word

    keyboard_nearby = {
        'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'sfcx', 'e': 'wrsd',
        'f': 'dgvc', 'g': 'fhtb', 'h': 'gjny', 'i': 'ujko', 'j': 'hkum',
        'k': 'jlim', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'etfd', 's': 'awedxz', 't': 'ryfg',
        'u': 'yihj', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tugh',
        'z': 'asx'
    }

    typo_type = random.choice(['swap', 'delete', 'double', 'insert', 'replace', 'vowel_swap'])

    if typo_type == 'swap' and len(word) >= 2:

        i = random.randint(0, len(word)-2)
        return word[:i] + word[i+1] + word[i] + word[i+2:]
    
    elif typo_type == 'delete' and len(word) >= 3:

        i = random.randint(0, len(word)-1)
        return word[:i] + word[i+1:]
    
    elif typo_type == 'double' and len(word) >= 2:

        i = random.randint(0, len(word)-1)
        return word[:i] + word[i] + word[i] + word[i+1:]
    
    elif typo_type == 'insert' and len(word) >= 2:

        i = random.randint(0, len(word))
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return word[:i] + char + word[i:]
    
    elif typo_type == 'replace' and len(word) >= 2:

        i = random.randint(0, len(word)-1)
        if word[i].lower() in keyboard_nearby:
            replacement = random.choice(keyboard_nearby[word[i].lower()])
            return word[:i] + replacement + word[i+1:]
        
        return word
    
    elif typo_type == 'vowel_swap' and len(word) >= 2:

        vowel_swaps = {'e': 'i', 'i': 'e', 'a': 'e', 'o': 'a'}
        vowel_positions = [j for j, c in enumerate(word) if c.lower() in vowel_swaps]

        if vowel_positions:
            i = random.choice(vowel_positions)
            new_vowel = vowel_swaps[word[i].lower()]
            
            return word[:i] + new_vowel + word[i+1:]
        
        return word

    return word
