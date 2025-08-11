import random
import math
from typing import List

from genetic_optimize.TFparamsBase import TFparamsBase

class Player:
    def __init__(self, player_id):
        self.id = player_id
        self.rating = 1000  # Initial ELO rating
        self.matches = set()  # Record opponent IDs that have been battled

    def __repr__(self):
        return f"Player {self.id} (ELO: {self.rating:.1f}, Skill: {self.true_skill:.1f})"

def elo_update(winner: TFparamsBase, loser: TFparamsBase, k=32):
    # ELO rating calculation formula
    expected = 1 / (1 + 10 ** ((loser.rating - winner.rating) / 400))
    delta = k * (1 - expected)
    winner.rating += delta
    loser.rating -= delta

def swiss_pairing(players: List[TFparamsBase]):
    # Sort by rating in descending order
    sorted_players = sorted(players, key=lambda x: -x.rating)
    paired = set()
    pairs = []

    for i in range(len(sorted_players)):
        if i in paired: continue
        
        # Find the nearest opponent that hasn't been battled
        for j in range(i+1, len(sorted_players)):
            if j in paired: continue
            if sorted_players[j].id not in sorted_players[i].matches:
                pairs.append((sorted_players[i], sorted_players[j]))
                paired.update([i, j])
                break
        else:  # Random matching when no available opponent
            for j in range(i+1, len(sorted_players)):
                if j not in paired:
                    pairs.append((sorted_players[i], sorted_players[j]))
                    paired.update([i, j])
                    break
    return pairs

def simulate_match(p1, p2):
    # Determine winner based on true skill (invisible in actual scenarios)
    if p1.true_skill > p2.true_skill:
        winner, loser = p1, p2
    elif p1.true_skill < p2.true_skill:
        winner, loser = p2, p1
    else:  # Handle draw
        winner, loser = random.choice([(p1, p2), (p2, p1)])
    
    # Update ELO rating
    elo_update(winner, loser)
    
    # Record battle history
    p1.matches.add(p2.id)
    p2.matches.add(p1.id)

def tournament(n=8):
    # Initialize players
    players = [Player(i) for i in range(n)]
    
    # Calculate required rounds (log2(n) + 2 rounds)
    rounds = int(math.log2(n)) + 2
    
    # Initial random pairing
    random.shuffle(players)
    for i in range(0, n, 2):
        p1, p2 = players[i], players[i+1]
        simulate_match(p1, p2)

    # Swiss round stage
    for _ in range(rounds-1):
        pairs = swiss_pairing(players)
        for p1, p2 in pairs:
            simulate_match(p1, p2)

    # Final ranking
    final_ranking = sorted(players, key=lambda x: (-x.rating, -x.true_skill))
    return final_ranking

# Run example
if __name__ == "__main__":
    final = tournament(8)
    print("Final ranking (ELO rating + true skill):")
    for i, p in enumerate(final, 1):
        print(f"{i:2}. {p}")