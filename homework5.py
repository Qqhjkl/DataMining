class Reasoner:
    def __init__(self, tbox):
        self.tbox = tbox

    def normalize(self):
        # 在本例中，TBox已经是规范化的
        return self.tbox

    def is_entailment(self, concept1, concept2):
        # 实现蕴涵检查
        normalized_tbox = self.normalize()
        # 使用规范化的TBox检查concept1是否被concept2包含
        return self.check_subsumption(normalized_tbox, concept1, concept2)

    def check_subsumption(self, tbox, concept1, concept2):
        # 检查概念1是否被概念2包含
        if concept1 == concept2:
            return True
        for axiom in tbox:
            if self.apply_axiom(axiom, concept1, concept2):
                return True
        return False

    def apply_axiom(self, axiom, concept1, concept2):
        # 应用一个公理来检查子类关系
        if len(axiom) == 3 and axiom[0] == concept1 and axiom[2] == concept2:
            return True
        elif len(axiom) == 4:
            a, b, r, c = axiom
            if concept1 == a and concept2 == c:
                # 检查是否存在b
                return self.check_subsumption(self.tbox, concept1, b)
        return False

# 定义TBox公理
tbox = [
    ('A', 'B', 'r', 'C'),
    ('C', 's', 'D'),
    ('exists(r, exists(s, top))', 'B', 'D')
]

# 用TBox实例化推理器
reasoner = Reasoner(tbox)

# 检查A是否被D包含
result = reasoner.is_entailment('A', 'D')
print(f"A 蕴涵 D 吗？ {'是' if result else '否'}")
