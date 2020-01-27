import random

class card_deck:
    card_deck = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 'J', 'Q', 'K'] * 4

    def __init__(self):
        pass

    def shuffle(self):
        random.shuffle(self.card_deck)

    def pop(self):
        self.card_deck.pop()

new_card = card_deck()

print(new_card.card_deck)

new_card.shuffle()

print(new_card.card_deck)


new_card.pop()
