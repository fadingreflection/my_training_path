---
name: joern-code-analysis
description: Use the shared Joern service to import, inspect, query, and export the Code Property Graph for exactly one application project.
---

# Joern Code Analysis Skill

## Purpose

Use Joern as the authoritative static code analysis and CPG extraction engine.

This skill operates on one project at a time and must never infer relationships between different projects.

## Environment contract

Joern server:
- HTTP endpoint: `http://codegraph-joern:8080` from the Docker network.
- Host endpoint: `http://127.0.0.1:8080`.
- Authentication is supplied via environment variables.
- Source roots are mounted read-only under `/projects/<project_id>/src`.
- Exports are written under `/exports/<project_id>/raw/neo4jcsv`.
- Project IDs use lowercase slug syntax: `[a-z0-9._-]+`.

## Required inputs

Before doing any project-specific operation, resolve:

- `project_id`
- source path `/projects/<project_id>/src`
- requested analysis goal

Do not operate on an implicit active project.

## Project isolation rules

1. Every Joern project name MUST equal `project_id`.
2. Never combine source directories from multiple projects in one import.
3. Never answer a project-specific question from another project's active CPG.
4. Before a query, explicitly switch to or load the requested project.
5. Export only to `/exports/<project_id>/...`.
6. Treat same filenames, symbols, namespaces, package names, or full names in different projects as unrelated.

## Standard workflow

### 1. Verify project

Confirm that `/projects/<project_id>/src` exists.

### 2. Inspect workspace

Query Joern workspace before importing.

Prefer an existing valid project if it corresponds to the same source snapshot.
If the source revision changed and consistency matters, rebuild the project.

### 3. Import

Use:

```scala
importCode("/projects/<project_id>/src", "<project_id>")
```

`importCode` creates the project, builds the CPG, loads it, and generates standard overlays.

For analyses that require Joern OSS data flow, ensure the data-flow layer exists:

```scala
run.ossdataflow
```

Do not claim data-flow reachability if the required layer was not generated.

### 4. Query

Prefer narrow traversals over dumping the full CPG.

Common examples:

Methods:

```scala
cpg.method.name.l
```

Method by full name:

```scala
cpg.method.fullName("<full_name>").l
```

Calls from a method:

```scala
cpg.method.fullName("<full_name>").call.name.l
```

Callers of a method:

```scala
cpg.method.fullName("<full_name>").caller.fullName.l
```

Callees:

```scala
cpg.method.fullName("<full_name>").callee.fullName.l
```

Source locations:

```scala
cpg.method.fullName("<full_name>")
  .map(m => (m.filename, m.lineNumber, m.lineNumberEnd))
  .l
```

AST:

```scala
cpg.method.fullName("<full_name>").ast.l
```

CFG:

```scala
cpg.method.fullName("<full_name>").cfgNode.l
```

### 5. Validate results

Before reporting a path:

- verify all nodes belong to the active project;
- preserve source file and line locations;
- distinguish direct Joern facts from agent inference;
- do not convert name similarity into a call edge;
- do not invent unresolved calls.

### 6. Export

For complete export use the infrastructure helper:

```bash
./scripts/export-project.sh <project_id>
```

Expected directory:

```text
/exports/<project_id>/raw/neo4jcsv
```

Joern supports full graph export with:

```bash
joern-export --repr=all --format=neo4jcsv
```

The raw CSV is not the stable agent schema. It must be normalized before long-term use in Neo4j.

## Canonical mapping guidance

Preserve Joern's original node type in `joern_label`, while mapping to one canonical node kind:

- FILE -> `File`
- NAMESPACE / NAMESPACE_BLOCK -> `Namespace`
- TYPE / TYPE_DECL -> `Type`
- METHOD -> `Method`
- METHOD_PARAMETER_IN / METHOD_PARAMETER_OUT -> `Parameter`
- LOCAL -> `Local`
- CALL -> `Call`
- IDENTIFIER -> `Identifier`
- LITERAL -> `Literal`
- CONTROL_STRUCTURE -> `ControlStructure`
- MEMBER -> `Member`
- METHOD_RETURN / RETURN -> `Return`
- anything else -> `Unknown`

Canonical edges include:

- AST -> `AST_CHILD`
- CALL -> `CALLS` only when the edge semantically represents invocation resolution
- ARGUMENT -> `ARGUMENT`
- CFG -> `CFG_NEXT`
- CDG -> `CDG_DEPENDS_ON`
- DDG / REACHING_DEF -> `DDG_DEPENDS_ON`
- REF -> `REFERS_TO`
- BINDS -> `BINDS`
- INHERITS_FROM -> `INHERITS_FROM`

Never discard the original Joern edge type; retain it as `joern_edge_type`.

## HTTP query pattern

POST JSON:

```json
{"query":"<CPGQL query>"}
```

to `/query-sync`.

Use authentication from environment variables.
Do not embed credentials in prompts, logs, generated Cypher, or reports.

## Failure handling

If parsing fails:
1. identify frontend/language;
2. return parser diagnostics;
3. do not silently fall back to regex-based code relationships.

If the project is absent:
1. import it;
2. validate the CPG;
3. then query it.

If server state is inconsistent or memory usage becomes abnormal:
1. preserve source and exports;
2. restart the Joern service;
3. re-import only the requested project.

## Output contract

For project-specific discoveries return structured facts where possible:

```json
{
  "project_id": "saleor",
  "facts": [
    {
      "type": "call_path",
      "source": "api.entry",
      "target": "product.tasks.fetch",
      "nodes": [],
      "evidence": [],
      "confidence": "direct"
    }
  ]
}
```

Confidence:
- `direct` — represented by Joern CPG edges/properties;
- `derived` — deterministic traversal over direct facts;
- `inferred` — agent inference, must be explicitly marked.

Never report `inferred` as if it were a Joern fact.
