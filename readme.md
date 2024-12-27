# Mini-LISP Interpreter
## Overview
Mini-LISP is a simple interpreter for a subset of the LISP programming language, implemented in Python. It supports essential LISP features, including arithmetic operations, boolean logic, conditionals, function definitions, recursion, and nested functions. Additionally, it includes type checking to ensure operations are performed on compatible data types.

## Structure
- Compiler Final Project.pdf
    - The Mini LISP proj description provided by the TAs
- MiniLisp.pdf
    - The Mini LISP proj detail of how to implement basic feature provided by the TAs
- public_test_data
    - The Mini LISP proj testing data provided by the TAs
- public_test_data_ans.pdf
    - The ansaer of Mini LISP proj testing data provided by the TAs
- mini_lisp.py
    - python code of implementing all the basic features
- recursion.py
    - base on mini_lisp.py, add bonus_feature_1: recursion
- nested.py
    - base on recursion.py, add bonus_feature_3: nested structure
- run_tests.py
    - script to run the public_test_data

## Code Explaination
以下為程式各個部分的功能與流程說明：

〈Token 類別〉
• 用來儲存字串轉換後最基本的語言元素（例如 '('、')'、數字、布林值、識別符號等）。
• 類別中包含 ttype（代表 Token 類型）和 value（代表實際值）。

〈tokenize 函式〉
• 將原始 Mini-LISP 程式碼 (code) 轉換為一串 Token。
• 根據程式碼中的字元，檢查是否為空白、括號，或符合數字/布林/識別符號的正則表達式來切分。
• 遇到不符合預期的字元會拋出 SyntaxError。

〈ASTNode 類別〉
• 建立抽象語法樹（Abstract Syntax Tree，AST）的節點。
• 透過 ntype（節點類型）、value（節點附帶的值）、children（子節點）描述程式的結構。
• 例如 ntype 可為 'number'、'bool'、'var'、'define'、'if'、'fun'、'call' 等，以代表此節點要做的事。

〈Parser 類別〉
• 接收 tokenize 後的 Token，依語法規則把 Token 解析成 AST。
• parse_program(): 讀取所有 Token，重複呼叫 parse_stmt() 建立整個程式的 AST。
• parse_stmt(): 判斷是否以左括號 '(' 開頭，如果是就進入 parse_sexpr()；否則 parse_exp()。
• parse_sexpr(): 處理各種小括號開頭的表達式，可能是 (define ...)、(print-num ...)、(if ...)、(fun ...)、或是一個函式呼叫。
• parse_fun_body(): 允許在函式內部先解析多個 define，再接一個最終表達式，以支援巢狀函式。
• parse_operator_or_func_call(): 區分內建運算子（+、-、*、/、mod、>、<、=、and、or、not）或自訂函式呼叫。

〈Environment 類別〉
• 用于儲存變數與函式的值。
• env_chain 是一個陣列，每個元素都是一個字典，模擬巢狀或區域作用域。
• define(name, value): 在最內層（頂端）定義/新增變數與值。
• set(name, value): 若變數在較外層作用域已有定義，則覆蓋；否則在最內層定義新值。
• lookup(name): 從內到外搜尋並回傳變數或函式的值；若找不到就拋出錯誤。

〈eval_ast 函式〉
• 根據 ASTNode 的類型 (ntype) 進行對應的求值。
• 若為 'number' or 'bool' or 'var'，直接回傳數字、布林、或在 Environment 中查到的值。
• 若為 'define'，先在 Environment 中放 placeholder（以便遞迴或函式可參考自己），再真正求值並設定該值。
• 若為 'print-num' 或 'print-bool'，先對子表達式求值，再印出對應格式。
• 若為 'if'，先求值條件，必須是布林，然後決定執行哪個分支。
• 若為 'fun'，回傳一個包含 'type' = 'function'、參數清單、函式本體、以及當前閉包環境的物件。
• 若為 'fun-body'，依序執行所有定義跟最後一個表達式，回傳最後一個結果。
• 若為 'call'，代表函式呼叫：
    - 先求值把要呼叫的函式以及參數都算出來。
    - 建立一個新的 Environment，複製函式儲存的環境，並 push() 一個新區域來綁定參數。
    - 執行函式本體 (fun-body)。
        • 內建運算子（+、-、*、/、mod、>、<、=、and、or、not）則呼叫 eval_operator 做進一步處理。
〈eval_operator 函式〉
• 接收運算子名稱 (op_name) 與子引數的 AST，先將每個子引數做 eval_ast，得到 Python 中對應值，再針對運算子做對應邏輯：
    - 數字運算: +、-、*、/、mod
    - 比較運算: >、<、=
    - 布林運算: and、or、not
• 若遇到不支援的運算子則拋出語法錯誤。

〈run_mini_lisp 函式〉
• 將整個流程串接：
    - tokenize：將原始字串變成 Token 清單。
    - parser.parse_program()：將 Token 清單解析成 AST。
    - 產生一個 Environment，逐一 eval_ast。
        • 若有錯誤則輸出 "syntax error"（或拋特定例外）。

整體而言，此程式先把字串轉成 Token，然後組成抽象語法樹 AST，最後在一個作用域控制的 Environment 裡面執行每個 ASTNode。這樣就能支援 Mini-LISP 的定義 (define)、函式 호출 (call)、條件式 (if)、內建運算子（+、-、*、等），以及巢狀函式和遞迴呼叫等進階特性。

## Usage
### Running the Interpreter
You can run Mini-LISP programs using the nested.py script.
```
python nested.py path/to/your_program.lsp
```
If no file is provided, a sample program will be executed.

## Testing
The project includes a run_tests.py script to execute all .lsp test files located in the public_test_data folder.

### Running Tests
```
python run_tests.py
```
Each test file will be executed, and the output will be displayed. Type errors and syntax errors will be reported accordingly.
