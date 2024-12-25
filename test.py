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

        # Comments? (If needed, not specified in the grammar, so ignoring)
        # if c == ';':
        #     while i < length and code[i] != '\n':
        #         i += 1
        #     continue

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
            # Otherwise, must be an ID or operator symbol
            else:
                # We'll treat everything else as an ID, but remain cautious:
                #   - Must match the grammar ID: letter (letter|digit|'-')*
                #   - Or it is an operator (like +, -, *, /, etc.)
                # Operators are reserved words, but let's allow them as ID tokens
                # and later interpret them in the parser. The grammar states
                # that certain tokens are "reserved words", e.g. define, fun, if, ...
                tokens.append(Token('ID', chunk))

            continue

        # Fallback: unrecognized character
        raise SyntaxError(f"syntax error: unrecognized character '{c}' at index {i}")

    return tokens


#################################################################
# PARSER: Build an AST from tokens
#################################################################

# We will parse according to the grammar:
#
# PROGRAM ::= STMT+
# STMT ::= EXP | DEF-STMT | PRINT-STMT
# PRINT-STMT ::= (print-num EXP) | (print-bool EXP)
# EXP ::= bool-val | number | VARIABLE | NUM-OP | LOGICAL-OP 
#         | FUN-EXP | FUN-CALL | IF-EXP
#
# and so on.

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

        # If it's a left paren, we parse what's inside
        if tk and tk.type == 'LPAREN':
            # We have something like (some-operator ...)
            # We'll parse a single s-expression
            expr_ast = self.parse_sexpr()

            # After parse_sexpr, we get an AST. We need to check if
            # it's a define or print or some expression, etc.
            # Actually parse_sexpr() will already differentiate if it's define, print, etc.
            # By grammar, (define ...) or (print-num ...) or (print-bool ...) or ...
            return expr_ast
        else:
            # If it's not '(', we expect a literal bool, number, or variable usage => an EXP
            return self.parse_exp()

    # def parse_sexpr(self):
    #     """
    #     Parse an s-expression that starts with '('.
    #     We'll then see the next token to figure out what it is.
    #     E.g.: (define ...), (print-num ...), (if ...), (fun ...), ...
    #     """
    #     self.match('LPAREN')  # consume '('
    #     head = self.peek()

    #     if not head:
    #         raise SyntaxError("syntax error: unexpected end after '('")

    #     # Check the head token
    #     if head.type == 'ID':
    #         # Could be define, print-num, print-bool, +, -, /, mod, ...
    #         # or a user-defined function name => function call
    #         op = head.value
    #         if op == "define":
    #             return self.parse_define()
    #         elif op == "print-num":
    #             return self.parse_print_stmt_num()
    #         elif op == "print-bool":
    #             return self.parse_print_stmt_bool()
    #         elif op == "if":
    #             return self.parse_if()
    #         elif op == "fun":
    #             return self.parse_fun_expr()
    #         else:
    #             # Possibly a builtin operator or a function call
    #             return self.parse_operator_or_func_call()

    #     else:
    #         # Unexpected: we do not expect '(' followed by a token type that is not ID
    #         raise SyntaxError(f"syntax error: unexpected token after '(' -> {head}")
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

        # Case 2: Otherwise, we might have an inline function call whose callee is an expression
        # Example: ((fun (x) (+ x 1)) 3)
        # We'll parse the first expression as the "callee".
        callee_expr = self.parse_exp()
        
        # Then parse zero or more expressions as parameters
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

        # Build an AST node 'call' with children: [callee_expr, param1, param2, ...]
        # We'll differentiate that from the named function call by leaving .value = None
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
        node = ASTNode('if', children=[test_expr, then_expr, else_expr])
        return node

    def parse_fun_expr(self):
        """
        FUN-EXP ::= (fun FUN_IDs FUN-BODY)
        FUN-IDs ::= (id*)
        """
        self.match('ID', 'fun')  # consume 'fun'

        # Next token must be '(' => the parameter list
        self.match('LPAREN')
        params = []
        while True:
            nxt = self.peek()
            if not nxt:
                raise SyntaxError("syntax error: unexpected end of tokens while parsing function params")
            if nxt.type == 'RPAREN':
                break
            # each param must be an ID
            t = self.match('ID')
            params.append(t.value)
        self.match('RPAREN')  # consume ')'

        # Now parse function body => an expression
        body_expr = self.parse_exp()

        self.match('RPAREN')  # consume the outer ')'
        # Build an AST node for the function
        node = ASTNode('fun', children=[body_expr], value=params)
        return node

    def parse_operator_or_func_call(self):
        """
        Could be a built-in operator ( +, -, *, /, mod, >, <, =, and, or, not )
        or a function call: (someFunc x y ... )
        We'll parse them in a uniform way, then differentiate based on the head symbol.
        """
        op_token = self.match('ID')  # consume operator or function name
        op_name = op_token.value

        # Distinguish recognized operators vs. function call
        recognized_ops = {
            '+', '-', '*', '/', 'mod', '>', '<', '=', 'and', 'or', 'not'
        }

        children = []
        while True:
            nxt = self.peek()
            if not nxt:
                raise SyntaxError("syntax error: unexpected end while parsing operator/func call")
            if nxt.type == 'RPAREN':
                # end of this s-expression
                break
            # parse next expression as a child
            child_expr = self.parse_exp()
            children.append(child_expr)

        self.match('RPAREN')  # consume ending ')'

        if op_name in recognized_ops:
            # We have a built-in operator
            return ASTNode(op_name, children=children)
        else:
            # We have a function call
            # ASTNode type 'call', value=op_name, children=children
            return ASTNode('call', value=op_name, children=children)

    def parse_exp(self):
        """
        EXP ::= bool-val | number | VARIABLE | NUM-OP | LOGICAL-OP 
                | FUN-EXP | FUN-CALL | IF-EXP
        But we handle them by looking at the current token.
        """
        tk = self.peek()
        if not tk:
            raise SyntaxError("syntax error: unexpected end of tokens in parse_exp")

        if tk.type == 'BOOL':
            self.match('BOOL')
            # tk.value is '#t' or '#f'
            val = True if tk.value == '#t' else False
            return ASTNode('bool', value=val)

        elif tk.type == 'NUMBER':
            self.match('NUMBER')
            val = int(tk.value)
            return ASTNode('number', value=val)

        elif tk.type == 'ID':
            # Could be a variable usage or an s-expression starting with '('
            # But if we see an ID at parse_exp (not inside parentheses),
            # then it's a variable (or we might confirm with the grammar).
            # Because if it's (ID ...) we parse_sexpr. If it's ID alone, it's a var usage.
            # Let's treat it as a variable usage
            self.match('ID')
            return ASTNode('var', value=tk.value)

        elif tk.type == 'LPAREN':
            # Then parse an s-expression
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
        # each element is a dict: {varName: value}
        self.env_chain = [{}]

    def push(self):
        self.env_chain.insert(0, {})

    def pop(self):
        self.env_chain.pop(0)

    def define(self, name, value):
        # define in the current (top) environment
        self.env_chain[0][name] = value

    def set(self, name, value):
        # set in existing environment if it exists
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
    """
    Evaluate an ASTNode in the given environment.
    Return the result as either a Python boolean (for #t/#f),
    an integer, or a function object (Python-level lambda or a custom representation).
    """
    ntype = node.ntype
    val = node.value
    children = node.children

    # Basic types
    if ntype == 'number':
        return val  # integer
    if ntype == 'bool':
        return val  # boolean True/False
    if ntype == 'var':
        # lookup variable in environment
        return env.lookup(val)

    # Statements
    if ntype == 'define':
        var_name = node.value
        expr_val = eval_ast(children[0], env)
        env.define(var_name, expr_val)
        # define statement returns None or we can return the value
        # Not strictly defined in the specs, but we'll return None
        return None

    if ntype == 'print-num':
        expr_val = eval_ast(children[0], env)
        if not isinstance(expr_val, int):
            raise TypeError(f"(print-num) expects integer, got {expr_val}")
        print(expr_val)
        return None

    if ntype == 'print-bool':
        expr_val = eval_ast(children[0], env)
        if not isinstance(expr_val, bool):
            raise TypeError(f"(print-bool) expects boolean, got {expr_val}")
        print("#t" if expr_val else "#f")
        return None

    # if expression
    if ntype == 'if':
        test_val = eval_ast(children[0], env)
        if not isinstance(test_val, bool):
            raise TypeError(f"(if) test expression must be boolean, got {test_val}")
        if test_val:
            return eval_ast(children[1], env)
        else:
            return eval_ast(children[2], env)

    # function creation
    if ntype == 'fun':
        # node.value => list of parameter names
        # node.children[0] => body expression
        param_names = node.value
        body_expr = children[0]

        # We'll represent the function as a Python dict:
        # {
        #   'type': 'function',
        #   'params': [...],
        #   'body': ASTNode,
        #   'env': (a snapshot of current environment)
        # }
        return {
            'type': 'function',
            'params': param_names,
            'body': body_expr,
            'env': [dict(scope) for scope in env.env_chain]  # shallow copy environment chain
        }

    # function call
    # if ntype == 'call':
    #     # node.value => function name
    #     # children => list of parameter expressions
    #     func_val = env.lookup(val)  # either a function dict or something else
    #     if not isinstance(func_val, dict) or func_val.get('type') != 'function':
    #         raise TypeError(f"Attempt to call a non-function: {val}")

    #     # Evaluate arguments
    #     arg_vals = [eval_ast(c, env) for c in children]
    if ntype == 'call':
        # If node.value is not None, it's a named function call
        if node.value is not None:
            func_val = env.lookup(node.value)
            arg_nodes = children
        else:
            # It's an anonymous inline function call: children[0] is the callee expression, 
            # children[1..] are the parameters
            callee_ast = children[0]
            func_val = eval_ast(callee_ast, env)  # Evaluate the function expression
            arg_nodes = children[1:]

        # Now 'func_val' should be a function dict
        if not (isinstance(func_val, dict) and func_val.get('type') == 'function'):
            raise TypeError("Attempt to call a non-function")
        
        # Evaluate arguments
        arg_vals = [eval_ast(arg_node, env) for arg_node in arg_nodes]

        # Execute function in a new environment derived from the function's closure
        func_env_chain = []
        # Rebuild environment from function definition time
        for scope in func_val['env']:
            func_env_chain.append(dict(scope))

        func_env = Environment()
        func_env.env_chain = func_env_chain

        # Push a new scope
        func_env.push()

        # Bind parameters
        params = func_val['params']
        if len(params) != len(arg_vals):
            raise TypeError(f"Function '{val}' expected {len(params)} args, got {len(arg_vals)}")

        for p, a in zip(params, arg_vals):
            func_env.define(p, a)

        # Evaluate body
        result = eval_ast(func_val['body'], func_env)
        return result

    # Built-in operators
    if ntype in ['+', '-', '*', '/', 'mod', '>', '<', '=', 'and', 'or', 'not']:
        return eval_operator(ntype, children, env)

    raise SyntaxError(f"syntax error: unknown AST node type '{ntype}'")


def eval_operator(op_name, children, env):
    # Evaluate all child expressions
    vals = [eval_ast(child, env) for child in children]

    # Numeric ops: +, -, *, /, mod
    if op_name == '+':
        # (+ EXP EXP+)
        # sum all
        s = 0
        for v in vals:
            if not isinstance(v, int):
                raise TypeError(f"(+) expects integers, got {v}")
            s += v
        return s

    if op_name == '-':
        # (- EXP EXP)
        # In the grammar, minus requires exactly 2 expressions.
        if len(vals) != 2:
            raise SyntaxError("(-) must have exactly 2 operands in this Mini-LISP subset")
        a, b = vals
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError(f"(-) expects integers, got {a}, {b}")
        return a - b

    if op_name == '*':
        # (* EXP EXP+)
        product = 1
        for v in vals:
            if not isinstance(v, int):
                raise TypeError(f"(*) expects integers, got {v}")
            product *= v
        return product

    if op_name == '/':
        # (/ EXP EXP)
        if len(vals) != 2:
            raise SyntaxError("(/) must have exactly 2 operands")
        a, b = vals
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError(f"(/) expects integers, got {a}, {b}")
        if b == 0:
            raise ZeroDivisionError("division by zero in (/)")
        return a // b  # integer division

    if op_name == 'mod':
        # (mod EXP EXP)
        if len(vals) != 2:
            raise SyntaxError("(mod) must have exactly 2 operands")
        a, b = vals
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError(f"(mod) expects integers, got {a}, {b}")
        if b == 0:
            raise ZeroDivisionError("division by zero in (mod)")
        return a % b

    # Comparison ops: >, <, =
    if op_name == '>':
        # (> EXP EXP)
        if len(vals) != 2:
            raise SyntaxError("(>) must have exactly 2 operands")
        a, b = vals
        if not (isinstance(a, int) and isinstance(b, int)):
            raise TypeError("(>) expects integers")
        return a > b

    if op_name == '<':
        # (< EXP EXP)
        if len(vals) != 2:
            raise SyntaxError("(<) must have exactly 2 operands")
        a, b = vals
        if not (isinstance(a, int) and isinstance(b, int)):
            raise TypeError("(<) expects integers")
        return a < b

    if op_name == '=':
        # (= EXP EXP+)
        # The grammar says: (= EXP EXP+). That means 2+ expressions?
        # We'll interpret that as "all are equal".
        if len(vals) < 2:
            raise SyntaxError("(=) must have at least 2 operands")
        # check all pairwise
        first = vals[0]
        if not all(isinstance(v, int) for v in vals):
            raise TypeError("(=) expects integers only")
        for v in vals[1:]:
            if v != first:
                return False
        return True

    # Logical ops: and, or, not
    if op_name == 'and':
        # (and EXP EXP+)
        if len(vals) < 2:
            raise SyntaxError("(and) must have at least 2 operands")
        for v in vals:
            if not isinstance(v, bool):
                raise TypeError("(and) expects booleans")
        return all(vals)

    if op_name == 'or':
        # (or EXP EXP+)
        if len(vals) < 2:
            raise SyntaxError("(or) must have at least 2 operands")
        for v in vals:
            if not isinstance(v, bool):
                raise TypeError("(or) expects booleans")
        return any(vals)

    if op_name == 'not':
        # (not EXP)
        if len(vals) != 1:
            raise SyntaxError("(not) must have exactly 1 operand")
        if not isinstance(vals[0], bool):
            raise TypeError("(not) expects a boolean")
        return not vals[0]

    # Should not reach here
    raise SyntaxError(f"Unknown operator '{op_name}'")


#################################################################
# DRIVER FUNCTION
#################################################################

def run_mini_lisp(code):
    """
    Main entry point to run a string of Mini-LISP code.
    Returns None; prints results of print statements to stdout.
    """
    # 1. Tokenize
    tokens = tokenize(code)

    # 2. Parse -> list of AST statements
    parser = Parser(tokens)
    program_ast = parser.parse_program()

    # 3. Evaluate each statement
    env = Environment()
    for stmt in program_ast:
        eval_ast(stmt, env)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            sample_program = file.read()
    else:
        sample_program = r"""
          (print-num((fun (x) (+ x 1)) 3))
        """
    run_mini_lisp(sample_program)
# Example usage (uncomment to try):
# if __name__ == "__main__":
#     sample_program = r"""
#         (print-bool #t)
#         (print-bool #f)

#         (print-bool (and #t #f))
#         (print-bool (and #t #t))

#         (print-bool (or #t #f))
#         (print-bool (or #f #f))

#         (print-bool (not #t))
#         (print-bool (not #f))


#     """
#     run_mini_lisp(sample_program)
