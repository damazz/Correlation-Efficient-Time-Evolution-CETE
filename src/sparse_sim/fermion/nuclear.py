

class Atom:
    symbol: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float

    def __init__(self, symbol: str, x: float, y: float, z: float, vx: float, vy: float, vz: float, mass: float):
        self.symbol = symbol
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass

    def evolve(self, fx, fy, fz, dt):
        ax = fx / self.mass
        ay = fy / self.mass
        az = fz / self.mass
        x = self.x + self.vx * dt + 0.5 * ax * dt**2
        y = self.y + self.vy * dt + 0.5 * ay * dt**2
        z = self.z + self.vz * dt + 0.5 * az * dt**2
        vx = self.vx + ax * dt
        vy = self.vy + ay * dt
        vz = self.vz + az * dt

        return Atom(self.symbol, x, y, z, vx, vy, vz, self.mass)

    def shift(self, dx, dy, dz):
        return Atom(self.symbol, self.x + dx, self.y + dy, self.z + dz, self.vx, self.vy, self.vz, self.mass)

    def energy(self):
        """Calculate the total energy of the molecule in Ha"""
        return 0.5 * self.mass * (self.vx**2 + self.vy**2 + self.vz**2)

    def to_string(self):
        return f"{self.symbol} {self.x} {self.y} {self.z}"

    def __str__(self):
        return self.to_string()


def interatomic_distance(atom1: Atom, atom2: Atom):
    dx = atom1.x - atom2.x
    dy = atom1.y - atom2.y
    dz = atom1.z - atom2.z
    return (dx**2 + dy**2 + dz**2) ** 0.5


class Molecule:
    atoms: list[Atom]
    symbol: str

    def __init__(self, symbol):
        self.symbol = symbol
        self.atoms = []

    def add_atom(self, atom: Atom):
        self.atoms.append(atom)

    def to_string(self):
        output = ""
        for i, atom in enumerate(self.atoms):
            output += f"{atom}"
            if i < len(self.atoms) - 1:
                output += "; "
        return output

    def __str__(self):
        return self.to_string()

    def evolve(self, forces, dt):
        new_mol = Molecule(self.symbol)
        for i, atom in enumerate(self.atoms):
            fx, fy, fz = forces[i]
            new_mol.add_atom(atom.evolve(fx, fy, fz, dt))
        return new_mol

    def shift_at_idx(self, idx, dx, dy, dz):
        new_mol = Molecule(self.symbol)
        for i, atom in enumerate(self.atoms):
            if i == idx:
                new_mol.add_atom(atom.shift(dx, dy, dz))
            else:
                new_mol.add_atom(atom)
        return new_mol

    def kinetic_energy(self):
        """Calculate the total energy of the molecule in Ha"""
        total_energy = 0
        for atom in self.atoms:
            total_energy += atom.energy()
        return total_energy
