from sparse_sim.fermion.nuclear import *


def test_atom():
    H_left = Atom("H", 0, 0, 0, 0, 0, 0, 1)

    print(H_left)


def test_mol():
    dist = 0.7

    mol0 = Molecule("H2")
    H_left = Atom("H", 0, 0, 0, 0, 0, 0, 1)
    H_right = Atom("H", 0, 0, dist, 0, 0, 0, 1)
    mol0.add_atom(H_left)
    mol0.add_atom(H_right)

    print(mol0)


def main():
    print("Testing Atom Class init and print")
    test_atom()

    print("\nTesting Molecule Class init and print")
    test_mol()


if __name__ == "__main__":
    main()
