# Noeta DSL - Complete System Flow Diagram

**Last Updated**: December 15, 2025

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        A[CLI Interface<br/>noeta_runner.py]
        B[Jupyter Kernel<br/>noeta_kernel.py]
    end

    subgraph "Compilation Pipeline"
        C[Lexer<br/>noeta_lexer.py]
        D[Parser<br/>noeta_parser.py]
        E[AST<br/>noeta_ast.py]
        F[Code Generator<br/>noeta_codegen.py]
    end

    subgraph "Runtime Environment"
        G[Python Interpreter]
        H[Pandas/NumPy/Matplotlib]
    end

    subgraph "Output Layer"
        I[Console Output]
        J[Jupyter Notebook Display]
        K[File Exports]
    end

    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    H --> J
    H --> K

    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#fff9c4
    style D fill:#fff9c4
    style E fill:#fff9c4
    style F fill:#fff9c4
    style G fill:#c8e6c9
    style H fill:#c8e6c9
    style I fill:#ffccbc
    style J fill:#ffccbc
    style K fill:#ffccbc
```

---

## 2. Detailed Compilation Pipeline Flow

```mermaid
flowchart TD
    START([User Input:<br/>Noeta Source Code])

    subgraph "PHASE 1: Lexical Analysis"
        LEX1[Initialize Lexer<br/>with source code]
        LEX2{Valid<br/>Characters?}
        LEX3[Identify Token Type:<br/>KEYWORD, IDENTIFIER,<br/>STRING, NUMBER, OPERATOR]
        LEX4[Create Token Object<br/>type, value, position]
        LEX5[Add to Token List]
        LEX6{More<br/>Characters?}
        LEX_ERR[Lexical Error:<br/>Invalid character]
    end

    subgraph "PHASE 2: Syntactic Analysis"
        PARSE1[Initialize Parser<br/>with tokens]
        PARSE2[Parse Statement]
        PARSE3{Statement<br/>Type?}
        PARSE4A[parse_load]
        PARSE4B[parse_filter]
        PARSE4C[parse_select]
        PARSE4D[parse_join]
        PARSE4E[parse_groupby]
        PARSE4F[parse_visualization]
        PARSE4G[Other parse methods...]
        PARSE5[Create AST Node<br/>LoadNode, FilterNode, etc.]
        PARSE6[Add to AST List]
        PARSE7{More<br/>Tokens?}
        PARSE_ERR[Syntax Error:<br/>Unexpected token]
    end

    subgraph "PHASE 3: Code Generation"
        GEN1[Initialize CodeGenerator<br/>symbol_table, imports]
        GEN2[Iterate through AST]
        GEN3{Node<br/>Type?}
        GEN4A[visit_LoadNode]
        GEN4B[visit_FilterNode]
        GEN4C[visit_SelectNode]
        GEN4D[visit_JoinNode]
        GEN4E[visit_GroupByNode]
        GEN4F[visit_PlotNode]
        GEN4G[Other visitors...]
        GEN5[Generate Python Code<br/>for operation]
        GEN6[Update symbol_table<br/>with new aliases]
        GEN7[Add required imports]
        GEN8{More<br/>Nodes?}
        GEN9[Assemble final code:<br/>imports + operations]
        GEN10{Plots<br/>used?}
        GEN11[Add plt.show]
    end

    subgraph "PHASE 4: Execution"
        EXEC1[Create execution<br/>namespace]
        EXEC2[exec Python code<br/>in namespace]
        EXEC3{Runtime<br/>Error?}
        EXEC_ERR[Runtime Error:<br/>Pandas/NumPy exception]
    end

    subgraph "PHASE 5: Output"
        OUT1{Output<br/>Type?}
        OUT2[Console print]
        OUT3[Matplotlib figure]
        OUT4[File export]
        OUT5[Jupyter display_data]
    end

    END([Output Delivered<br/>to User])

    START --> LEX1
    LEX1 --> LEX2
    LEX2 -->|Yes| LEX3
    LEX2 -->|No| LEX_ERR
    LEX3 --> LEX4
    LEX4 --> LEX5
    LEX5 --> LEX6
    LEX6 -->|Yes| LEX2
    LEX6 -->|No| PARSE1

    PARSE1 --> PARSE2
    PARSE2 --> PARSE3
    PARSE3 --> PARSE4A
    PARSE3 --> PARSE4B
    PARSE3 --> PARSE4C
    PARSE3 --> PARSE4D
    PARSE3 --> PARSE4E
    PARSE3 --> PARSE4F
    PARSE3 --> PARSE4G
    PARSE4A --> PARSE5
    PARSE4B --> PARSE5
    PARSE4C --> PARSE5
    PARSE4D --> PARSE5
    PARSE4E --> PARSE5
    PARSE4F --> PARSE5
    PARSE4G --> PARSE5
    PARSE5 --> PARSE6
    PARSE6 --> PARSE7
    PARSE7 -->|Yes| PARSE2
    PARSE7 -->|No| GEN1
    PARSE3 -.->|Invalid| PARSE_ERR

    GEN1 --> GEN2
    GEN2 --> GEN3
    GEN3 --> GEN4A
    GEN3 --> GEN4B
    GEN3 --> GEN4C
    GEN3 --> GEN4D
    GEN3 --> GEN4E
    GEN3 --> GEN4F
    GEN3 --> GEN4G
    GEN4A --> GEN5
    GEN4B --> GEN5
    GEN4C --> GEN5
    GEN4D --> GEN5
    GEN4E --> GEN5
    GEN4F --> GEN5
    GEN4G --> GEN5
    GEN5 --> GEN6
    GEN6 --> GEN7
    GEN7 --> GEN8
    GEN8 -->|Yes| GEN2
    GEN8 -->|No| GEN9
    GEN9 --> GEN10
    GEN10 -->|Yes| GEN11
    GEN10 -->|No| EXEC1
    GEN11 --> EXEC1

    EXEC1 --> EXEC2
    EXEC2 --> EXEC3
    EXEC3 -->|Yes| EXEC_ERR
    EXEC3 -->|No| OUT1

    OUT1 --> OUT2
    OUT1 --> OUT3
    OUT1 --> OUT4
    OUT1 --> OUT5
    OUT2 --> END
    OUT3 --> END
    OUT4 --> END
    OUT5 --> END

    LEX_ERR -.->|Report| END
    PARSE_ERR -.->|Report| END
    EXEC_ERR -.->|Report| END

    style START fill:#4caf50,color:#fff
    style END fill:#4caf50,color:#fff
    style LEX_ERR fill:#f44336,color:#fff
    style PARSE_ERR fill:#f44336,color:#fff
    style EXEC_ERR fill:#f44336,color:#fff
```

---

## 3. CLI Execution Flow

```mermaid
flowchart TD
    START([User runs:<br/>python noeta_runner.py])

    CMD1{Command<br/>Type?}

    subgraph "File Execution Path"
        FILE1[Parse file path<br/>from sys.argv]
        FILE2{File<br/>exists?}
        FILE3[Read file contents]
        FILE_ERR[Error: File not found]
    end

    subgraph "Inline Code Path"
        INLINE1[Parse -c flag<br/>from sys.argv]
        INLINE2{Code<br/>provided?}
        INLINE3[Extract code string]
        INLINE_ERR[Error: No code provided]
    end

    subgraph "Compilation"
        COMP1[compile_noeta<br/>source_code]
        COMP2[Lexer.tokenize]
        COMP3[Parser.parse]
        COMP4[CodeGenerator.generate]
        COMP5[Return Python code string]
        COMP_ERR[Compilation Error]
    end

    subgraph "Execution"
        EXEC1[execute_noeta<br/>source_code, verbose]
        EXEC2{Verbose<br/>mode?}
        EXEC3[Print generated<br/>Python code]
        EXEC4[exec Python code<br/>in globals]
        EXEC5[Print output]
        EXEC_ERR[Execution Error]
    end

    END([Exit with<br/>status code])

    START --> CMD1
    CMD1 -->|File path| FILE1
    CMD1 -->|-c flag| INLINE1

    FILE1 --> FILE2
    FILE2 -->|Yes| FILE3
    FILE2 -->|No| FILE_ERR
    FILE3 --> COMP1

    INLINE1 --> INLINE2
    INLINE2 -->|Yes| INLINE3
    INLINE2 -->|No| INLINE_ERR
    INLINE3 --> COMP1

    COMP1 --> COMP2
    COMP2 --> COMP3
    COMP3 --> COMP4
    COMP4 --> COMP5
    COMP5 --> EXEC1
    COMP2 -.->|Error| COMP_ERR
    COMP3 -.->|Error| COMP_ERR
    COMP4 -.->|Error| COMP_ERR

    EXEC1 --> EXEC2
    EXEC2 -->|Yes| EXEC3
    EXEC2 -->|No| EXEC4
    EXEC3 --> EXEC4
    EXEC4 --> EXEC5
    EXEC4 -.->|Error| EXEC_ERR
    EXEC5 --> END

    FILE_ERR -.-> END
    INLINE_ERR -.-> END
    COMP_ERR -.-> END
    EXEC_ERR -.-> END

    style START fill:#2196f3,color:#fff
    style END fill:#4caf50,color:#fff
    style FILE_ERR fill:#f44336,color:#fff
    style INLINE_ERR fill:#f44336,color:#fff
    style COMP_ERR fill:#f44336,color:#fff
    style EXEC_ERR fill:#f44336,color:#fff
```

---

## 4. Jupyter Kernel Flow

```mermaid
flowchart TD
    START([User executes cell<br/>in Jupyter Notebook])

    subgraph "Kernel Initialization"
        INIT1[install_kernel.py<br/>registers kernel spec]
        INIT2[Jupyter loads<br/>NoetaKernel class]
        INIT3[Initialize kernel<br/>with banner, version]
    end

    subgraph "Code Execution"
        EXEC1[do_execute receives<br/>code from cell]
        EXEC2[Increment execution_count]
        EXEC3[compile_noeta<br/>source_code]
        EXEC4{Compilation<br/>Success?}
        EXEC5[exec Python code<br/>in self.namespace]
        EXEC6{Execution<br/>Success?}
        EXEC_ERR[Send error message<br/>to notebook]
    end

    subgraph "Output Handling"
        OUT1{Output<br/>Type?}
        OUT2[Capture stdout/stderr]
        OUT3[Send stream output]
        OUT4[Check for matplotlib<br/>figures]
        OUT5{Figure<br/>exists?}
        OUT6[Convert figure to PNG]
        OUT7[Base64 encode]
        OUT8[Send display_data<br/>with image/png]
        OUT9[Clear figure]
    end

    subgraph "Interactive Features"
        INT1[do_complete:<br/>provide autocomplete]
        INT2[Return Noeta keywords<br/>and operations]
        INT3[do_inspect:<br/>provide help text]
    end

    subgraph "Response"
        RESP1[Build execution_reply]
        RESP2[Include status:<br/>ok/error/abort]
        RESP3[Include execution_count]
    end

    END([Notebook displays<br/>results])

    START --> EXEC1
    EXEC1 --> EXEC2
    EXEC2 --> EXEC3
    EXEC3 --> EXEC4
    EXEC4 -->|Yes| EXEC5
    EXEC4 -->|No| EXEC_ERR
    EXEC5 --> EXEC6
    EXEC6 -->|Yes| OUT1
    EXEC6 -->|No| EXEC_ERR

    OUT1 --> OUT2
    OUT1 --> OUT4
    OUT2 --> OUT3
    OUT3 --> RESP1
    OUT4 --> OUT5
    OUT5 -->|Yes| OUT6
    OUT5 -->|No| RESP1
    OUT6 --> OUT7
    OUT7 --> OUT8
    OUT8 --> OUT9
    OUT9 --> RESP1

    EXEC_ERR --> RESP1

    RESP1 --> RESP2
    RESP2 --> RESP3
    RESP3 --> END

    START -.->|Tab pressed| INT1
    INT1 --> INT2
    INT2 --> END

    START -.->|Shift+Tab| INT3
    INT3 --> END

    INIT1 -.->|One-time setup| INIT2
    INIT2 -.-> INIT3

    style START fill:#2196f3,color:#fff
    style END fill:#4caf50,color:#fff
    style EXEC_ERR fill:#f44336,color:#fff
    style INIT1 fill:#ff9800,color:#fff
```

---

## 5. Symbol Table Management Flow

```mermaid
flowchart LR
    subgraph "Operation 1: LOAD"
        L1[load data.csv as df1]
        L2[visit_LoadNode]
        L3[Generate:<br/>df1 = pd.read_csv]
        L4[symbol_table:<br/>df1 -> df1]
    end

    subgraph "Operation 2: FILTER"
        F1[filter df1<br/>where age > 25 as df2]
        F2[visit_FilterNode]
        F3[Resolve df1<br/>from symbol_table]
        F4[Generate:<br/>df2 = df1[df1.age > 25]]
        F5[symbol_table:<br/>df1 -> df1<br/>df2 -> df2]
    end

    subgraph "Operation 3: SELECT"
        S1[select name, age<br/>from df2 as df3]
        S2[visit_SelectNode]
        S3[Resolve df2<br/>from symbol_table]
        S4[Generate:<br/>df3 = df2[[name, age]]]
        S5[symbol_table:<br/>df1 -> df1<br/>df2 -> df2<br/>df3 -> df3]
    end

    L1 --> L2 --> L3 --> L4
    L4 --> F1
    F1 --> F2 --> F3 --> F4 --> F5
    F5 --> S1
    S1 --> S2 --> S3 --> S4 --> S5

    style L4 fill:#fff9c4
    style F5 fill:#fff9c4
    style S5 fill:#fff9c4
```

---

## 6. Operation Category Flow

```mermaid
graph TD
    START([Noeta Operation])

    subgraph "Data I/O"
        IO1[LOAD:<br/>CSV -> DataFrame]
        IO2[SAVE:<br/>DataFrame -> CSV]
    end

    subgraph "Transformation"
        T1[SELECT:<br/>Column projection]
        T2[FILTER:<br/>Row filtering]
        T3[SORT:<br/>Ordering]
        T4[MUTATE:<br/>Column creation]
        T5[APPLY:<br/>Lambda functions]
    end

    subgraph "Aggregation"
        A1[GROUPBY:<br/>Group + aggregate]
        A2[SAMPLE:<br/>Random sampling]
    end

    subgraph "Cleaning"
        C1[DROPNA:<br/>Remove nulls]
        C2[FILLNA:<br/>Impute values]
    end

    subgraph "Joining"
        J1[JOIN:<br/>Merge DataFrames]
    end

    subgraph "Analysis"
        AN1[DESCRIBE:<br/>Statistics]
        AN2[SUMMARY:<br/>DataFrame info]
        AN3[OUTLIERS:<br/>IQR detection]
        AN4[QUANTILE:<br/>Percentiles]
        AN5[NORMALIZE:<br/>Min-max scaling]
        AN6[BINNING:<br/>Discretization]
        AN7[ROLLING:<br/>Window functions]
        AN8[HYPOTHESIS:<br/>T-test]
    end

    subgraph "Visualization"
        V1[BOXPLOT]
        V2[HEATMAP]
        V3[PAIRPLOT]
        V4[TIMESERIES]
        V5[PIE]
        V6[EXPORT_PLOT]
    end

    END([Generated<br/>Python/Pandas Code])

    START --> IO1 & IO2
    START --> T1 & T2 & T3 & T4 & T5
    START --> A1 & A2
    START --> C1 & C2
    START --> J1
    START --> AN1 & AN2 & AN3 & AN4 & AN5 & AN6 & AN7 & AN8
    START --> V1 & V2 & V3 & V4 & V5 & V6

    IO1 & IO2 --> END
    T1 & T2 & T3 & T4 & T5 --> END
    A1 & A2 --> END
    C1 & C2 --> END
    J1 --> END
    AN1 & AN2 & AN3 & AN4 & AN5 & AN6 & AN7 & AN8 --> END
    V1 & V2 & V3 & V4 & V5 & V6 --> END

    style START fill:#2196f3,color:#fff
    style END fill:#4caf50,color:#fff
```

---

## 7. Error Handling Flow

```mermaid
flowchart TD
    START([Input: Noeta Code])

    subgraph "Lexical Phase"
        L1[Lexer processes<br/>characters]
        L2{Valid<br/>tokens?}
        L_ERR[LexicalError:<br/>Invalid character at<br/>position X]
    end

    subgraph "Syntax Phase"
        P1[Parser processes<br/>tokens]
        P2{Valid<br/>grammar?}
        P_ERR[SyntaxError:<br/>Unexpected token 'X'<br/>at position Y]
    end

    subgraph "Semantic Phase"
        S1[Code Generator<br/>resolves symbols]
        S2{Symbol<br/>exists?}
        S_ERR[SemanticError:<br/>Undefined variable 'X']
    end

    subgraph "Runtime Phase"
        R1[Execute Python code]
        R2{Runtime<br/>error?}
        R_ERR[RuntimeError:<br/>Pandas/NumPy exception]
    end

    subgraph "Error Reporting"
        E1[Capture exception]
        E2[Format error message]
        E3{Execution<br/>Context?}
        E4[Print to stderr<br/>CLI]
        E5[Send error stream<br/>Jupyter]
    end

    SUCCESS([Successful<br/>Execution])

    START --> L1
    L1 --> L2
    L2 -->|Yes| P1
    L2 -->|No| L_ERR

    P1 --> P2
    P2 -->|Yes| S1
    P2 -->|No| P_ERR

    S1 --> S2
    S2 -->|Yes| R1
    S2 -->|No| S_ERR

    R1 --> R2
    R2 -->|No| SUCCESS
    R2 -->|Yes| R_ERR

    L_ERR --> E1
    P_ERR --> E1
    S_ERR --> E1
    R_ERR --> E1

    E1 --> E2
    E2 --> E3
    E3 -->|CLI| E4
    E3 -->|Jupyter| E5

    style START fill:#2196f3,color:#fff
    style SUCCESS fill:#4caf50,color:#fff
    style L_ERR fill:#f44336,color:#fff
    style P_ERR fill:#f44336,color:#fff
    style S_ERR fill:#f44336,color:#fff
    style R_ERR fill:#f44336,color:#fff
```

---

## 8. Import Management Flow

```mermaid
flowchart TD
    START([Code Generation<br/>Starts])

    INIT[Initialize:<br/>imports = set<br/>Add base imports]

    BASE[Default imports:<br/>pandas, numpy,<br/>matplotlib, seaborn, scipy]

    subgraph "Operation Processing"
        OP1{Node<br/>Type?}

        OP2[Data I/O Operation]
        OP3[Statistical Operation]
        OP4[Visualization Operation]
        OP5[ML Operation]

        IMP1[No additional<br/>imports needed]
        IMP2[Add scipy.stats]
        IMP3[Add seaborn plots]
        IMP4[Add sklearn modules]
    end

    subgraph "Import Assembly"
        ASM1[Iterate through<br/>imports set]
        ASM2[Generate import<br/>statements]
        ASM3[Prepend to<br/>generated code]
    end

    END([Python code with<br/>all required imports])

    START --> INIT
    INIT --> BASE
    BASE --> OP1

    OP1 --> OP2
    OP1 --> OP3
    OP1 --> OP4
    OP1 --> OP5

    OP2 --> IMP1
    OP3 --> IMP2
    OP4 --> IMP3
    OP5 --> IMP4

    IMP1 & IMP2 & IMP3 & IMP4 --> ASM1
    ASM1 --> ASM2
    ASM2 --> ASM3
    ASM3 --> END

    style START fill:#2196f3,color:#fff
    style END fill:#4caf50,color:#fff
    style BASE fill:#fff9c4
```

---

## 9. Complete End-to-End Example Flow

```mermaid
flowchart TD
    START([User writes Noeta code:<br/>load sales.csv as sales<br/>filter sales where revenue > 1000<br/>describe sales])

    subgraph "LEXER"
        LEX1[Token: LOAD]
        LEX2[Token: STRING sales.csv]
        LEX3[Token: AS]
        LEX4[Token: IDENTIFIER sales]
        LEX5[Token: FILTER]
        LEX6[Token: IDENTIFIER sales]
        LEX7[Token: WHERE]
        LEX8[Token: IDENTIFIER revenue]
        LEX9[Token: OPERATOR >]
        LEX10[Token: NUMBER 1000]
        LEX11[Token: DESCRIBE]
        LEX12[Token: IDENTIFIER sales]
    end

    subgraph "PARSER"
        PARSE1[LoadNode:<br/>file=sales.csv<br/>alias=sales]
        PARSE2[FilterNode:<br/>source=sales<br/>condition=revenue>1000<br/>alias=sales]
        PARSE3[DescribeNode:<br/>source=sales]
    end

    subgraph "CODE GENERATOR"
        GEN1[visit_LoadNode:<br/>sales = pd.read_csv<br/>symbol_table[sales] = sales]
        GEN2[visit_FilterNode:<br/>sales = sales[sales.revenue > 1000]<br/>symbol_table[sales] = sales]
        GEN3[visit_DescribeNode:<br/>print sales.describe]
    end

    PYTHON[Generated Python:<br/>import pandas as pd<br/>...<br/>sales = pd.read_csv 'sales.csv'<br/>sales = sales[sales['revenue'] > 1000]<br/>print sales.describe]

    EXEC[Execute Python code:<br/>Load CSV -> DataFrame<br/>Filter rows<br/>Print statistics]

    OUTPUT[Console Output:<br/>count  mean  std  min  25%  50%  75%  max<br/>...]

    END([User sees results])

    START --> LEX1
    LEX1 --> LEX2 --> LEX3 --> LEX4
    LEX4 --> LEX5 --> LEX6 --> LEX7
    LEX7 --> LEX8 --> LEX9 --> LEX10
    LEX10 --> LEX11 --> LEX12

    LEX12 --> PARSE1
    PARSE1 --> PARSE2
    PARSE2 --> PARSE3

    PARSE3 --> GEN1
    GEN1 --> GEN2
    GEN2 --> GEN3

    GEN3 --> PYTHON
    PYTHON --> EXEC
    EXEC --> OUTPUT
    OUTPUT --> END

    style START fill:#2196f3,color:#fff
    style END fill:#4caf50,color:#fff
    style PYTHON fill:#fff9c4
```

---

## 10. System Component Interaction Diagram

```mermaid
graph TB
    subgraph "User Layer"
        U1[User<br/>CLI Command]
        U2[User<br/>Jupyter Cell]
    end

    subgraph "Interface Layer"
        I1[noeta_runner.py<br/>CLI Entry Point]
        I2[noeta_kernel.py<br/>Jupyter Interface]
    end

    subgraph "Compilation Layer"
        C1[noeta_lexer.py<br/>Tokenization]
        C2[noeta_parser.py<br/>Syntax Analysis]
        C3[noeta_ast.py<br/>AST Definitions]
        C4[noeta_codegen.py<br/>Code Generation]
    end

    subgraph "Execution Layer"
        E1[Python Interpreter<br/>exec]
        E2[Namespace<br/>globals/locals]
    end

    subgraph "Library Layer"
        L1[Pandas<br/>DataFrames]
        L2[NumPy<br/>Arrays]
        L3[Matplotlib<br/>Plotting]
        L4[Seaborn<br/>Visualization]
        L5[SciPy<br/>Statistics]
        L6[scikit-learn<br/>ML]
    end

    subgraph "Output Layer"
        O1[Console<br/>stdout/stderr]
        O2[Files<br/>CSV/PNG]
        O3[Jupyter<br/>display_data]
    end

    U1 --> I1
    U2 --> I2

    I1 --> C1
    I2 --> C1

    C1 --> C2
    C2 --> C3
    C3 --> C4

    C4 --> E1
    E1 --> E2

    E2 --> L1
    E2 --> L2
    E2 --> L3
    E2 --> L4
    E2 --> L5
    E2 --> L6

    L1 --> O1
    L2 --> O1
    L3 --> O2
    L4 --> O2
    L5 --> O1
    L6 --> O1

    I2 --> O3
    O2 --> O3

    style U1 fill:#e3f2fd
    style U2 fill:#e3f2fd
    style I1 fill:#bbdefb
    style I2 fill:#bbdefb
    style C1 fill:#fff9c4
    style C2 fill:#fff9c4
    style C3 fill:#fff9c4
    style C4 fill:#fff9c4
    style E1 fill:#c8e6c9
    style E2 fill:#c8e6c9
    style L1 fill:#b2dfdb
    style L2 fill:#b2dfdb
    style L3 fill:#b2dfdb
    style L4 fill:#b2dfdb
    style L5 fill:#b2dfdb
    style L6 fill:#b2dfdb
    style O1 fill:#ffccbc
    style O2 fill:#ffccbc
    style O3 fill:#ffccbc
```

---

## Key Decision Points Summary

| Phase | Decision Point | Criteria | Outcome |
|-------|---------------|----------|---------|
| **Input** | Execution Mode | CLI flag (-c) or file path | File read vs. inline code |
| **Lexer** | Token Type | Character pattern matching | Token categorization |
| **Parser** | Statement Type | First keyword token | Appropriate parse_* method |
| **Code Gen** | Node Type | AST node class | Appropriate visit_* method |
| **Code Gen** | Plot Detection | Visualization node present | Add plt.show() or not |
| **Execution** | Runtime Error | Exception raised | Continue or abort |
| **Output** | Context Type | CLI vs. Jupyter | Console print vs. display_data |
| **Output** | Content Type | stdout vs. figure | Stream vs. image encoding |

---

## Notes for Stakeholders

1. **Modularity**: Each phase is independent and can be enhanced without affecting others
2. **Extensibility**: New operations require additions to all four compilation components
3. **Error Handling**: Multi-layer error detection with clear reporting
4. **Performance**: Direct Python code generation ensures near-native Pandas performance
5. **Interoperability**: Works seamlessly with existing Python data science ecosystem
6. **Debugging**: Verbose mode shows generated Python code for transparency
7. **User Experience**: Both CLI and notebook interfaces for different workflows
