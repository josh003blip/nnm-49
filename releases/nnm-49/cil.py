"""CIL v1: recursive expressions compiled to a bounded binary stack machine.

program := (program (def name expr)* expr)
expr := number | x | name | (concept expr expr) | (focus expr)
      | (add expr expr) | (mul expr expr) | (learn expr)
      | (rel weight tag[,tag]* expr expr)
concept evaluates its focus (second child); focus is identity. rel returns
weight * (target - source), with epistemic attenuation. Definitions are
compiled once as shared executable subroutines, not expanded text.
"""
import math
import re
import struct

TAGS = {'uncertain': 1, 'contradictory': 2, 'incomplete': 4,
        'hypothetical': 8, 'supported': 16}
MAGIC = b'CIL\x01'
CONST, INPUT, ADD, MUL, LEARN, CONCEPT, FOCUS, REL, CALL, RET = range(10)

def parse(source):
    tokens = re.findall(r'\(|\)|[^\s()]+', source)
    pos = 0
    def read(depth=0):
        nonlocal pos
        if depth > 100 or pos >= len(tokens):
            raise ValueError('CIL nesting/truncation limit')
        token = tokens[pos]; pos += 1
        if token == ')': raise ValueError('unexpected closing parenthesis')
        if token != '(': return token
        result = []
        while pos < len(tokens) and tokens[pos] != ')': result.append(read(depth+1))
        if pos == len(tokens): raise ValueError('unclosed expression')
        pos += 1
        return result
    tree = read()
    if pos != len(tokens): raise ValueError('trailing tokens')
    return tree

def compile_cil(source):
    tree = parse(source)
    if not isinstance(tree, list) or len(tree) < 2 or tree[0] != 'program':
        raise ValueError('expected program')
    blocks = []; names = {}
    def emit(expr):
        if isinstance(expr, str):
            if expr == 'x': return bytes([INPUT])
            if expr in names: return bytes([CALL]) + struct.pack('<H', names[expr])
            try: value = float(expr)
            except ValueError: raise ValueError('unknown symbol: ' + expr)
            if not math.isfinite(value): raise ValueError('nonfinite constant')
            return bytes([CONST]) + struct.pack('<d', value)
        if not expr: raise ValueError('empty expression')
        op, *args = expr
        arities = {'add': 2, 'mul': 2, 'learn': 1, 'concept': 2, 'focus': 1, 'rel': 4}
        if op not in arities or len(args) != arities[op]: raise ValueError('invalid operation/arity')
        if op == 'rel':
            weight = float(args[0])
            if not math.isfinite(weight): raise ValueError('nonfinite weight')
            mask = 0
            for tag in args[1].split(','):
                if tag not in TAGS: raise ValueError('unknown epistemic tag')
                mask |= TAGS[tag]
            return emit(args[2])+emit(args[3])+bytes([REL])+struct.pack('<dB', weight, mask)
        code = b''.join(emit(arg) for arg in args)
        return code + bytes([{'add': ADD, 'mul': MUL, 'learn': LEARN,
                             'concept': CONCEPT, 'focus': FOCUS}[op]])
    for definition in tree[1:-1]:
        if not isinstance(definition, list) or len(definition) != 3 or definition[0] != 'def':
            raise ValueError('invalid definition')
        name = definition[1]
        if not isinstance(name, str) or name in names or name == 'x': raise ValueError('invalid name')
        block = emit(definition[2]) + bytes([RET])
        names[name] = len(blocks); blocks.append(block)
    blocks.append(emit(tree[-1]) + bytes([RET]))
    return MAGIC + struct.pack('<H', len(blocks)) + b''.join(struct.pack('<I', len(b))+b for b in blocks)

def execute(binary, x, core, budget=10000):
    """Decode and execute binary only; source trees are never evaluated."""
    if binary[:4] != MAGIC: raise ValueError('invalid CIL version')
    try:
        count, = struct.unpack_from('<H', binary, 4); pos = 6; blocks = []
        for _ in range(count):
            size, = struct.unpack_from('<I', binary, pos); pos += 4
            if size == 0 or pos + size > len(binary): raise ValueError('invalid block size')
            blocks.append(binary[pos:pos+size]); pos += size
        if pos != len(binary) or not blocks: raise ValueError('invalid binary framing')
        remaining = budget
        def run(index, depth=0):
            nonlocal remaining
            if depth > 100: raise ValueError('call depth exceeded')
            code = blocks[index]; pc = 0; stack = []
            while pc < len(code):
                remaining -= 1
                if remaining < 0: raise ValueError('instruction budget exceeded')
                op = code[pc]; pc += 1
                if op == CONST:
                    v, = struct.unpack_from('<d', code, pc); pc += 8; stack.append(v)
                elif op == INPUT: stack.append(float(x))
                elif op == CALL:
                    target, = struct.unpack_from('<H', code, pc); pc += 2
                    if target >= index: raise ValueError('non-acyclic call')
                    stack.append(run(target, depth+1))
                elif op == LEARN: stack.append(float(core.predict(stack.pop())))
                elif op == FOCUS: stack.append(stack.pop())
                elif op in (ADD, MUL, CONCEPT, REL):
                    b = stack.pop(); a = stack.pop()
                    if op == ADD: stack.append(a+b)
                    elif op == MUL: stack.append(a*b)
                    elif op == CONCEPT: stack.append(b)
                    else:
                        weight, mask = struct.unpack_from('<dB', code, pc); pc += 9
                        if mask & ~31: raise ValueError('invalid tag bits')
                        attenuation = 0 if mask & 2 else .5 if mask & 13 else 1
                        stack.append(weight * attenuation * (b-a))
                elif op == RET:
                    if len(stack) != 1 or pc != len(code): raise ValueError('invalid return')
                    return stack[0]
                else: raise ValueError('unknown opcode')
            raise ValueError('missing return')
        result = run(len(blocks)-1)
        if not math.isfinite(result): raise ValueError('nonfinite result')
        return result
    except (struct.error, IndexError, OverflowError) as error:
        raise ValueError('malformed CIL binary') from error
