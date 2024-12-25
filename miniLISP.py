import re
import operator

class MiniLispError(Exception):
    pass

def tokenize(source):
    # Breaks the source into tokens (parentheses, symbols, numbers)
    tokens = re.findall(r'\(|\)|[^\s()]+', source)
    return tokens

def parse(tokens):
    # Parse tokens into an AST
    if not tokens:
        raise MiniLispError("syntax error")
    token = tokens.pop(0)

    if token == '(':
        ast = []
        while tokens and tokens[0] != ')':
            ast.append(parse(tokens))
        if not tokens:
            raise MiniLispError("syntax error")
        tokens.pop(0)  # remove ')'
        return ast
    elif token == ')':
        raise MiniLispError("syntax error")
    else:
        return atom(token)

def atom(token):
    if token == "#t":
        return True
    elif token == "#f":
        return False
    try:
        return int(token)
    except ValueError:
        return 
    
def bool_to_sym(b):
    return "#t" if b else "#f"

def evaluate(ast, env):
    if isinstance(ast, int):
        return ast

    if isinstance(ast, str):
        # Variable reference
        if ast in env:
            return env[ast]
        else:
            raise MiniLispError(f"Undefined variable {ast}")

    if not isinstance(ast, list):
        raise MiniLispError("syntax error")

    if not ast:
        raise MiniLispError("syntax error")

    head = ast[0]

    # ----------------------------
    # Feature 2: print-num
    # (print-num expr)
    if head == 'print-num':
        if len(ast) != 2:
            raise MiniLispError("syntax error")
        val = evaluate(ast[1], env)
        if not isinstance(val, int):
            raise MiniLispError("syntax error")
        print(val)
        return val

    # ----------------------------
    # Feature 3: numerical operations
    # + - * / % (mod) etc.
    elif head in ['+', '-', '*', '/', 'mod']:
        if len(ast) < 3:
            raise MiniLispError("syntax error")
        ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.floordiv,
            'mod': operator.mod
        }
        result = evaluate(ast[1], env)
        if not isinstance(result, int):
            raise MiniLispError("syntax error")
        for arg in ast[2:]:
            val = evaluate(arg, env)
            if not isinstance(val, int):
                raise MiniLispError("syntax error")
            result = ops[head](result, val)
        return result

    # ----------------------------
    # Feature 4: logical operations
    # == > >= < <= and or not 
    elif head in ['==', '>', '>=', '<', '<=', 'and', 'or', 'not']:
        if head == 'not':
            if len(ast) != 2:
                raise MiniLispError("syntax error")
            val = bool(evaluate(ast[1], env))
            return bool_to_sym(not val)

        if len(ast) < 3:
            raise MiniLispError("syntax error")

        left = bool(evaluate(ast[1], env))
        right = bool(evaluate(ast[2], env))

        if head == '==':
            return bool_to_sym(left == right)
        elif head == '>':
            return bool_to_sym(evaluate(ast[1], env) > evaluate(ast[2], env))
        elif head == '>=':
            return bool_to_sym(evaluate(ast[1], env) >= evaluate(ast[2], env))
        elif head == '<':
            return bool_to_sym(evaluate(ast[1], env) < evaluate(ast[2], env))
        elif head == '<=':
            return bool_to_sym(evaluate(ast[1], env) <= evaluate(ast[2], env))
        elif head == 'and':
            return bool_to_sym(left and right)
        elif head == 'or':
            return bool_to_sym(left or right)
    # ----------------------------
    # Feature 5: if expression
    # (if condition true-branch false-branch)
    elif head == 'if':
        if len(ast) != 4:
            raise MiniLispError("syntax error")
        cond = evaluate(ast[1], env)
        if cond:
            return evaluate(ast[2], env)
        else:
            return evaluate(ast[3], env)

    # ----------------------------
    # Feature 6: define variable
    # (define var expr)
    elif head == 'define':
        if len(ast) != 3:
            raise MiniLispError("syntax error")
        var = ast[1]
        val = evaluate(ast[2], env)
        env[var] = val
        return val

    # ----------------------------
    # Feature 7: anonymous function
    # (lambda (args...) body)
    elif head == 'lambda':
        if len(ast) != 3:
            raise MiniLispError("syntax error")
        args = ast[1]
        body = ast[2]
        if not isinstance(args, list):
            raise MiniLispError("syntax error")
        return lambda *call_args: evaluate(
            body, {**env, **dict(zip(args, call_args))}
        )

    # ----------------------------
    # Feature 8: named function
    # (function fname (args...) body)
    elif head == 'function':
        if len(ast) != 4:
            raise MiniLispError("syntax error")
        fname = ast[1]
        args = ast[2]
        body = ast[3]
        if not isinstance(args, list):
            raise MiniLispError("syntax error")
        func = lambda *call_args: evaluate(
            body, {**env, **dict(zip(args, call_args))}
        )
        env[fname] = func
        return fname

    else:
        # Function call
        func = evaluate(head, env)
        if callable(func):
            evaluated_args = [evaluate(a, env) for a in ast[1:]]
            return func(*evaluated_args)
        else:
            raise MiniLispError("syntax error")

def repl():
    env = {}
    while True:
        try:
            source = input('mini-lisp> ')
            if not source.strip():
                continue
            tokens = tokenize(source)
            ast = parse(tokens)
            result = evaluate(ast, env)
        except MiniLispError as e:
            print("syntax error")
        except EOFError:
            break
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    repl()