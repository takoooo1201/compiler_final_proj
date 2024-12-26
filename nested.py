import sys
import re

#################################################################
# LEXER
#################################################################

class Token:
    def __init__(self, ttype, value):
        self.type = ttype  # e.g. 'NUMBER', 'BOOL', 'LPAREN', ...
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"


def tokenize(code):
    """
    Convert input code (string) into a list of Tokens.
    Raises SyntaxError if something unexpected is found.
    """
    tokens = []
    i = 0
    length = len(code)

    # Regex patterns for recognized tokens
    number_pattern = re.compile(r"^-?(0|[1-9]\d*)$")
    bool_pattern = re.compile(r"^#[tf]$")
    id_pattern = re.compile(r"^[a-z]([a-z0-9-]*)$")  # letter (letter|digit|'-')*

    while i < length:
        c = code[i]

        # Skip separators
        if c in [' ', '\t', '\n', '\r']:
            i += 1
            continue

        # Parentheses
        if c == '(':
            tokens.append(Token('LPAREN', '('))
            i += 1
            continue
        elif c == ')':
            tokens.append(Token('RPAREN', ')'))
            i += 1
            continue

        # If it's a symbol, number, bool, or ID
        # We read a chunk until the next whitespace/parenthesis
        if c not in ['(', ')', ' ', '\t', '\n', '\r']:
            start = i
            while i < length and code[i] not in ['(', ')', ' ', '\t', '\n', '\r']:
                i += 1
            chunk = code[start:i]

            # Check if chunk is a bool
            if bool_pattern.match(chunk):
                tokens.append(Token('BOOL', chunk))
            # Check if chunk is a number
            elif number_pattern.match(chunk):
                tokens.append(Token('NUMBER', chunk))
            else:
                # We'll treat everything else as an ID or operator symbol
                tokens.append(Token('ID', chunk))

            continue

        # Fallback: unrecognized character
        raise SyntaxError(f"syntax error: unrecognized character '{c}' at index {i}")

    return tokens


#################################################################
# PARSER: Build an AST from tokens
#################################################################

class ASTNode:
    def __init__(self, ntype, value=None, children=None):
        """
        ntype: e.g. 'number', 'bool', 'var', 'print-num', 'print-bool',
               '+', '-', '*', '/', 'mod', '>', '<', '=', 'and', 'or',
               'not', 'if', 'define', 'fun', 'call', ...
        value: sometimes storing literal value, e.g. numeric or bool or var name
        children: list of ASTNode children
        """
        self.ntype = ntype
        self.value = value
        self.children = children if children else []

    def __repr__(self):
        return f"ASTNode({self.ntype}, {self.value}, children={self.children})"


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.length = len(tokens)

    def current_token(self):
        if self.pos < self.length:
            return self.tokens[self.pos]
        return None

    def match(self, ttype=None, tvalue=None):
        """
        Return current token if it matches conditions.
        Otherwise raise SyntaxError.
        If ttype is None, we skip type check.
        If tvalue is None, we skip value check.
        """
        tk = self.current_token()
        if not tk:
            raise SyntaxError("syntax error: unexpected end of tokens")

        if ttype is not None and tk.type != ttype:
            raise SyntaxError(f"syntax error: expected token type '{ttype}' but got '{tk.type}'")
        if tvalue is not None and tk.value != tvalue:
            raise SyntaxError(f"syntax error: expected token value '{tvalue}' but got '{tk.value}'")

        self.pos += 1
        return tk

    def peek(self):
        """Return current token (or None) without consuming it."""
        if self.pos < self.length:
            return self.tokens[self.pos]
        return None

    def parse_program(self):
        """
        PROGRAM ::= STMT+
        We parse until we run out of tokens. Return a list of ASTNodes.
        """
        statements = []
        while self.pos < self.length:
            statements.append(self.parse_stmt())
        return statements

    def parse_stmt(self):
        """
        STMT ::= EXP | DEF-STMT | PRINT-STMT
        We'll look ahead: if it's '(' we might parse something inside.
        """
        tk = self.peek()
        if tk and tk.type == 'LPAREN':
            # parse a single s-expression
            return self.parse_sexpr()
        else:
            # parse a literal (bool, number) or a variable => an EXP
            return self.parse_exp()

    def parse_sexpr(self):
        """
        Parse an s-expression that starts with '('.
        We'll then see the next token to figure out what it is.
        E.g.: (define ...), (print-num ...), (if ...), (fun ...), ...
        
        Also covers calling an anonymous function:
            ((fun (x) (+ x 1)) 3)
        """
        self.match('LPAREN')  # consume '('
        head = self.peek()
        if not head:
            raise SyntaxError("syntax error: unexpected end after '('")

        # Case 1: If head.type == 'ID', check for define, print, if, fun, etc.
        if head.type == 'ID':
            if head.value == "define":
                return self.parse_define()
            elif head.value == "print-num":
                return self.parse_print_stmt_num()
            elif head.value == "print-bool":
                return self.parse_print_stmt_bool()
            elif head.value == "if":
                return self.parse_if()
            elif head.value == "fun":
                return self.parse_fun_expr()
            else:
                # Possibly a builtin operator or a named function call
                return self.parse_operator_or_func_call()

        # Case 2: Otherwise, an inline function call whose callee is an expression
        callee_expr = self.parse_exp()
        params = []
        while True:
            nxt = self.peek()
            if not nxt:
                raise SyntaxError("syntax error: unexpected end while parsing function call params")
            if nxt.type == 'RPAREN':
                break
            param_expr = self.parse_exp()
            params.append(param_expr)

        self.match('RPAREN')  # consume ')'
        # 'call' node with children: [callee_expr, param1, param2, ...], value=None
        return ASTNode('call', value=None, children=[callee_expr] + params)

    def parse_print_stmt_num(self):
        """
        PRINT-STMT ::= (print-num EXP)
        """
        self.match('ID', 'print-num')  # consume 'print-num'
        expr_node = self.parse_exp()
        self.match('RPAREN')
        return ASTNode('print-num', children=[expr_node])

    def parse_print_stmt_bool(self):
        """
        PRINT-STMT ::= (print-bool EXP)
        """
        self.match('ID', 'print-bool')  # consume 'print-bool'
        expr_node = self.parse_exp()
        self.match('RPAREN')
        return ASTNode('print-bool', children=[expr_node])

    def parse_define(self):
        """
        DEF-STMT ::= (define VARIABLE EXP)
        """
        self.match('ID', 'define')  # consume 'define'
        var_token = self.match('ID')  # variable name
        var_name = var_token.value
        expr_node = self.parse_exp()
        self.match('RPAREN')
        return ASTNode('define', value=var_name, children=[expr_node])

    def parse_if(self):
        """
        IF-EXP ::= (if TEST-EXP THEN-EXP ELSE-EXP)
        """
        self.match('ID', 'if')  # consume 'if'
        test_expr = self.parse_exp()
        then_expr = self.parse_exp()
        else_expr = self.parse_exp()
        self.match('RPAREN')
        return ASTNode('if', children=[test_expr, then_expr, else_expr])

    def parse_fun_expr(self):
        """
        FUN-EXP ::= (fun FUN_IDs FUN-BODY)
        FUN-IDs ::= (id*)
        fun-body ::= def-stmt* exp
        """
        self.match('ID', 'fun')  # consume 'fun'
        self.match('LPAREN')
        params = []
        while True:
            tk = self.peek()
            if tk and tk.type == 'ID':
                # collect parameter names
                param_token = self.match('ID')
                params.append(param_token.value)
            else:
                break
        self.match('RPAREN')  # consume ')'

        # Parse the function body, which may contain define statements
        body_node = self.parse_fun_body()
        # consume outer ')'
        self.match('RPAREN')
        return ASTNode('fun', children=[body_node], value=params)

    def parse_fun_body(self):
        """
        fun-body ::= def-stmt* exp
        Parse zero or more define statements followed by a single expression.
        We'll store them in a 'fun-body' node whose children are [def1, def2, ..., finalExpr].
        """
        body_children = []
        # Collect define statements first
        while True:
            tk = self.peek()
            if tk and tk.type == 'LPAREN':
                # Look ahead to see if it's (define ...)
                next_t = self.tokens[self.pos+1] if (self.pos+1 < self.length) else None
                if next_t and next_t.type == 'ID' and next_t.value == 'define':
                    define_node = self.parse_sexpr()  # parse (define ...)
                    body_children.append(define_node)
                    continue
            break

        # Now parse the final expression
        final_expr = self.parse_exp()
        body_children.append(final_expr)

        return ASTNode('fun-body', children=body_children)

    def parse_operator_or_func_call(self):
        """
        Could be a built-in operator ( +, -, *, /, mod, >, <, =, and, or, not )
        or a function call: (someFunc x y ... )
        """
        op_token = self.match('ID')
        op_name = op_token.value

        recognized_ops = {
            '+', '-', '*', '/', 'mod', '>', '<', '=', 'and', 'or', 'not'
        }

        children = []
        while True:
            nxt = self.peek()
            if not nxt:
                raise SyntaxError("syntax error: unexpected end while parsing operator/func call")
            if nxt.type == 'RPAREN':
                break
            child_expr = self.parse_exp()
            children.append(child_expr)

        self.match('RPAREN')  # consume ')'

        if op_name in recognized_ops:
            return ASTNode(op_name, children=children)
        else:
            # Named function call
            return ASTNode('call', value=op_name, children=children)

    def parse_exp(self):
        tk = self.peek()
        if not tk:
            raise SyntaxError("syntax error: unexpected end of tokens in parse_exp")

        if tk.type == 'BOOL':
            self.match('BOOL')
            val = True if tk.value == '#t' else False
            return ASTNode('bool', value=val)

        elif tk.type == 'NUMBER':
            self.match('NUMBER')
            val = int(tk.value)
            return ASTNode('number', value=val)

        elif tk.type == 'ID':
            self.match('ID')
            return ASTNode('var', value=tk.value)

        elif tk.type == 'LPAREN':
            return self.parse_sexpr()
        else:
            raise SyntaxError(f"syntax error: unexpected token in parse_exp -> {tk}")


#################################################################
# EVALUATOR
#################################################################

class Environment:
    """
    Environment to store variable bindings and function definitions.
    We keep it as a list of dictionaries (like a chain) so that
    function calls can have their own local environment easily.
    """
    def __init__(self):
        self.env_chain = [{}]

    def push(self):
        self.env_chain.insert(0, {})

    def pop(self):
        self.env_chain.pop(0)

    def define(self, name, value):
        self.env_chain[0][name] = value

    def set(self, name, value):
        for scope in self.env_chain:
            if name in scope:
                scope[name] = value
                return
        # if not found, define in top scope
        self.define(name, value)

    def lookup(self, name):
        for scope in self.env_chain:
            if name in scope:
                return scope[name]
        raise NameError(f"Undefined variable or function '{name}'")


def eval_ast(node, env):
    ntype = node.ntype
    val = node.value
    children = node.children

    # Basic types
    if ntype == 'number':
        return val
    if ntype == 'bool':
        return val
    if ntype == 'var':
        return env.lookup(val)

    # define
    if ntype == 'define':
        var_name = node.value
        # 1. put a placeholder so that the function can reference itself if needed
        env.define(var_name, None)
        # 2. evaluate the expression
        expr_val = eval_ast(children[0], env)
        # 3. update environment with the real value (function or otherwise)
        env.set(var_name, expr_val)
        return None

    # print-num
    if ntype == 'print-num':
        expr_val = eval_ast(children[0], env)
        if not isinstance(expr_val, int):
            raise TypeError(f"(print-num) expects integer, got {expr_val}")
        print(expr_val)
        return None

    # print-bool
    if ntype == 'print-bool':
        expr_val = eval_ast(children[0], env)
        if not isinstance(expr_val, bool):
            raise TypeError(f"(print-bool) expects boolean, got {expr_val}")
        print("#t" if expr_val else "#f")
        return None

    # if
    if ntype == 'if':
        test_val = eval_ast(children[0], env)
        if not isinstance(test_val, bool):
            raise TypeError(f"(if) test expression must be boolean, got {test_val}")
        if test_val:
            return eval_ast(children[1], env)
        else:
            return eval_ast(children[2], env)

    # fun
    if ntype == 'fun':
        param_names = node.value
        body_expr = children[0]
        return {
            'type': 'function',
            'params': param_names,
            'body': body_expr,
            # capture environment at definition time (closure)
            #'env': [dict(scope) for scope in env.env_chain]
            'env': env.env_chain[:] 
        }
     # function body containing define statements + final expression
    if ntype == 'fun-body':
        # Evaluate each child in order; last child's value is the function result.
        result = None
        for child in children:
            result = eval_ast(child, env)
        return result

    # call
    if ntype == 'call':
        if val is not None:
            # named function call
            func_val = env.lookup(val)
            arg_nodes = children
        else:
            # anonymous inline function call
            callee_ast = children[0]
            func_val = eval_ast(callee_ast, env)
            arg_nodes = children[1:]

        if not (isinstance(func_val, dict) and func_val.get('type') == 'function'):
            raise TypeError("Attempt to call a non-function")

        # Evaluate arguments
        arg_vals = [eval_ast(arg_node, env) for arg_node in arg_nodes]

        # Build a new environment from the function's closure
        func_env = Environment()
        func_env.env_chain = [dict(scope) for scope in func_val['env']]
        func_env.push()  # new local scope

        # Bind parameters
        params = func_val['params']
        if len(params) != len(arg_vals):
            raise TypeError(f"Function expected {len(params)} args, got {len(arg_vals)}")

        for p, a in zip(params, arg_vals):
            func_env.define(p, a)

        # Evaluate function body
        return eval_ast(func_val['body'], func_env)

    # built-in operators
    if ntype in ['+', '-', '*', '/', 'mod', '>', '<', '=', 'and', 'or', 'not']:
        return eval_operator(ntype, children, env)

    raise SyntaxError(f"syntax error: unknown AST node type '{ntype}'")


def eval_operator(op_name, children, env):
    vals = [eval_ast(child, env) for child in children]

    # + 
    if op_name == '+':
        if not vals:
            raise SyntaxError("(+) cannot have zero operands.")
        total = 0
        for v in vals:
            if not isinstance(v, int):
                raise TypeError(f"(+) expects integers, got {v}")
            total += v
        return total

    # -
    if op_name == '-':
        if len(vals) != 2:
            raise SyntaxError("(-) must have exactly 2 operands")
        a, b = vals
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError(f"(-) expects integers, got {a}, {b}")
        return a - b

    # *
    if op_name == '*':
        if not vals:
            raise SyntaxError("(*) cannot have zero operands.")
        product = 1
        for v in vals:
            if not isinstance(v, int):
                raise TypeError(f"(*) expects integers, got {v}")
            product *= v
        return product

    # /
    if op_name == '/':
        if len(vals) != 2:
            raise SyntaxError("(/) must have exactly 2 operands")
        a, b = vals
        if not (isinstance(a, int) and isinstance(b, int)):
            raise TypeError(f"(/) expects integers, got {a}, {b}")
        if b == 0:
            raise ZeroDivisionError("division by zero in (/)")
        return a // b

    # mod
    if op_name == 'mod':
        if len(vals) != 2:
            raise SyntaxError("(mod) must have exactly 2 operands")
        a, b = vals
        if not (isinstance(a, int) and isinstance(b, int)):
            raise TypeError(f"(mod) expects integers, got {a}, {b}")
        if b == 0:
            raise ZeroDivisionError("division by zero in (mod)")
        return a % b

    # >
    if op_name == '>':
        if len(vals) != 2:
            raise SyntaxError("(>) must have exactly 2 operands")
        a, b = vals
        if not (isinstance(a, int) and isinstance(b, int)):
            raise TypeError("(>) expects integers")
        return a > b

    # <
    if op_name == '<':
        if len(vals) != 2:
            raise SyntaxError("(<) must have exactly 2 operands")
        a, b = vals
        if not (isinstance(a, int) and isinstance(b, int)):
            raise TypeError("(<) expects integers")
        return a < b

    # =
    if op_name == '=':
        if len(vals) < 2:
            raise SyntaxError("(=) must have at least 2 operands")
        if not all(isinstance(v, int) for v in vals):
            raise TypeError("(=) expects integers only")
        first = vals[0]
        for v in vals[1:]:
            if v != first:
                return False
        return True

    # and
    if op_name == 'and':
        if len(vals) < 2:
            raise SyntaxError("(and) must have at least 2 operands")
        if not all(isinstance(v, bool) for v in vals):
            raise TypeError("(and) expects booleans")
        return all(vals)

    # or
    if op_name == 'or':
        if len(vals) < 2:
            raise SyntaxError("(or) must have at least 2 operands")
        if not all(isinstance(v, bool) for v in vals):
            raise TypeError("(or) expects booleans")
        return any(vals)

    # not
    if op_name == 'not':
        if len(vals) != 1:
            raise SyntaxError("(not) must have exactly 1 operand")
        if not isinstance(vals[0], bool):
            raise TypeError("(not) expects a boolean")
        return not vals[0]

    raise SyntaxError(f"Unknown operator '{op_name}'")


#################################################################
# DRIVER FUNCTION
#################################################################

def run_mini_lisp(code):
    """
    Main entry point to run a string of Mini-LISP code.
    """
    try:
        # 1. Tokenize
        tokens = tokenize(code)

        # 2. Parse -> list of AST statements
        parser = Parser(tokens)
        program_ast = parser.parse_program()

        # 3. Evaluate each statement
        env = Environment()
        for stmt in program_ast:
            eval_ast(stmt, env)
    except (SyntaxError, ValueError, TypeError, NameError) as e: 
        # The spec requires "syntax error" for any parse/eval error
        print("syntax error")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], 'r') as file:
                sample_program = file.read()
            run_mini_lisp(sample_program)
        except SyntaxError:
            print("syntax error")
    else:
        sample_program = r"""
          (print-num((fun (x) (+ x 1)) 3))
        """
        try:
            run_mini_lisp(sample_program)
        except SyntaxError:
            print("syntax error")
# Demo: Recursion Example
# if __name__ == "__main__":
#     sample_program = r"""
#         (define f
#             (fun (x)
#               (if (= x 1)
#                   1
#                   (* x (f (- x 1))))))
#         (print-num (f 4))

        
#     """

#     run_mini_lisp(sample_program)
