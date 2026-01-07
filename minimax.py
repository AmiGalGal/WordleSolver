import pandas as pd
from collections import defaultdict

#loads words
df = pd.read_csv("merged.csv")
WORDS = df["word"].str.lower().tolist()

#using the domain of solutions and of guesses to be diff because i can guess stuff i already tested
CANDIDATES = WORDS.copy()
GUESSES = WORDS.copy()

#feedback - turn the guess into a readable results
def feedback(guess, target):
    result = [0]*5
    target = list(target)

    for i in range(5):
        if guess[i] == target[i]:
            result[i] = 2
            target[i] = None

    for i in range(5):
        if result[i] == 0 and guess[i] in target:
            result[i] = 1
            target[target.index(guess[i])] = None

    return tuple(result)

#which candidates survive the elimination
def filter_candidates(candidates, guess, fb):
    return [
        w for w in candidates
        if feedback(guess, w) == fb
    ]

#what will remain after a guess, the minimax value
def worst_case_remaining(guess, candidates):
    buckets = defaultdict(int)
    for w in candidates:
        buckets[feedback(guess, w)] += 1
    return max(buckets.values())

#choosing the one that will minimize the max
def choose_best_guess(candidates, guesses):
    best_guess = None
    best_score = float("inf")

    for g in guesses:
        score = worst_case_remaining(g, candidates)
        if score < best_score:
            best_score = score
            best_guess = g

    return best_guess, best_score

#just an algorithm to see how it would solve known word, for testing purposes
def solve_wordle(secret, max_steps=6):
    candidates = CANDIDATES.copy()

    print(f"\nSecret: {secret}")

    for step in range(1, max_steps+1):
        if len(candidates) == 1:
            guess = candidates[0]
            score = 0
        else:
            guess, score = choose_best_guess(candidates, GUESSES)
        fb = feedback(guess, secret)

        print(f"Guess {step}: {guess} → {fb} (worst-case {score})")

        if all(x == 2 for x in fb):
            print(f"Solved in {step} guesses!")
            return step

        candidates = filter_candidates(candidates, guess, fb)
        print(f"Remaining candidates: {len(candidates)}")

    print("I have Failed nigga")
    return None

def play_wordle():
    candidates = CANDIDATES.copy()

    print("\nWordle Solver")
    print("Enter feedback after each guess:")
    print("2 = green, 1 = yellow, 0 = gray")
    print("Example: 20110\n")

    for turn in range(1, 7):
        if turn==1: #i just ran it one time to get the best 1st guess since its not affect by the word
            guess = "serai"
            score = 0
        elif len(candidates) == 1:
            guess = candidates[0]
        else:
            guess, score = choose_best_guess(candidates, GUESSES)
        print(f"Guess {turn}/6: {guess}")

        fb_str = input("Enter feedback (5 digits 0/1/2): ").strip()
        if len(fb_str) != 5 or any(c not in "012" for c in fb_str):
            print("Invalid input. Try again.")
            return

        fb = tuple(int(c) for c in fb_str)

        if all(x == 2 for x in fb):
            print(f"\nLets fucking gooo solved in {turn} guesses")
            return

        candidates = filter_candidates(candidates, guess, fb)
        print(f"Remaining candidates: {len(candidates)}\n")
        print(candidates)
        if len(candidates) == 0:
            print("No candidates left — check your shlomper.")
            return

    print("\nI have Failed nigga")


play_wordle()
