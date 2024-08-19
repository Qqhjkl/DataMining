class Reasoner:
    def __init__(self, tbox):
        self.tbox = tbox

    def normalize(self):
        return self.tbox

    def is_entailment(self, concept1, concept2):
        normalized_tbox = self.normalize()
        return self.check_subsumption(normalized_tbox, concept1, concept2)

    def check_subsumption(self, tbox, concept1, concept2):
        if concept1 == concept2:
            return True
        visited = set()
        return self.recursive_subsumption_check(tbox, concept1, concept2, visited)

    def recursive_subsumption_check(self, tbox, concept1, concept2, visited):
        if concept1 == concept2:
            return True
        if (concept1, concept2) in visited:
            return False
        visited.add((concept1, concept2))
        for axiom in tbox:
            if self.apply_axiom(axiom, concept1, concept2, visited):
                return True
        return False

    def apply_axiom(self, axiom, concept1, concept2, visited):
        if len(axiom) == 4:
            a, b, r, c = axiom
            if concept1 == a:
                # A -> B and B -> concept2
                if self.recursive_subsumption_check(self.tbox, b, concept2, visited):
                    return True
                # A -> ∃r.C and C -> concept2
                if self.recursive_subsumption_check(self.tbox, c, concept2, visited):
                    return True
        elif len(axiom) == 3:
            exists, b, d = axiom
            if 'exists' in exists and b == concept1:
                return self.recursive_subsumption_check(self.tbox, d, concept2, visited)
        return False

# 定义TBox公理
tbox = [
    ('A', 'B', 'r', 'C'),
    ('C', 's', 'D'),
    ('exists(r, exists(s, top))', 'B', 'D')
]


reasoner = Reasoner(tbox)

result = reasoner.is_entailment('A', 'D')
print(f"A 包含于 D 吗？ {'是' if result else '否'}")
