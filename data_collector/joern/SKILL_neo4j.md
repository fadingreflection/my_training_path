---
name: neo4j-code-graph
description: Store, validate, query, and enrich the canonical per-application code graph in Neo4j without creating cross-project relationships.
---

# Neo4j Code Graph Skill

## Purpose

Use Neo4j as the persistent graph workspace for an AI agent.

The graph stores normalized Joern CPG facts plus agent-created analytical entities and edges, but only within one application project.

## Core model

Every canonical code node carries:

```text
:CodeNode
project_id
uid
node_id
kind
schema_version
source
```

and one semantic label such as:

```text
File
Namespace
Type
Method
Parameter
Local
Call
Identifier
Literal
ControlStructure
Member
Return
Unknown
```

Example:

```cypher
(:CodeNode:Method {
  project_id: "saleor",
  uid: "saleor:1234",
  node_id: "1234",
  kind: "Method",
  name: "fetch_product_media_image_task",
  full_name: "...",
  file_path: "saleor/product/tasks.py",
  line_start: 440,
  line_end: 450,
  source: "joern",
  schema_version: "1"
})
```

## Canonical relationship types

Use only the stable canonical vocabulary for code relations:

```text
CONTAINS
DECLARES
AST_CHILD
CALLS
ARGUMENT
RETURNS
CFG_NEXT
CDG_DEPENDS_ON
DDG_DEPENDS_ON
REFERS_TO
BINDS
INHERITS_FROM
IMPLEMENTS
HAS_TYPE
```

Preserve Joern's original edge name in the relationship property `joern_edge_type`.

Agent-created analytical relationships must use a separate controlled vocabulary and `source="agent"`.

Do not overload a structural Joern edge with an agent inference.

## Project isolation invariant

For every relationship:

```text
start.project_id = relationship.project_id = end.project_id
```

Never create or merge relationships across projects.

The following is forbidden:

```cypher
MATCH (a:Method {full_name: $name}),
      (b:Method {full_name: $name})
MERGE (a)-[:SAME_AS]->(b)
```

because it can match different projects.

Always bind project scope:

```cypher
MATCH (a:Method {project_id: $project_id, full_name: $source}),
      (b:Method {project_id: $project_id, full_name: $target})
MERGE (a)-[:CALLS {project_id: $project_id}]->(b)
```

## Required query parameter

All project-specific Cypher must use `$project_id`.

Do not interpolate project IDs into raw query strings when driver parameters are available.

## Schema bootstrap

Create:

```cypher
CREATE CONSTRAINT code_node_uid IF NOT EXISTS
FOR (n:CodeNode)
REQUIRE n.uid IS UNIQUE;

CREATE INDEX code_node_project IF NOT EXISTS
FOR (n:CodeNode)
ON (n.project_id);

CREATE INDEX method_full_name IF NOT EXISTS
FOR (n:Method)
ON (n.project_id, n.full_name);

CREATE INDEX file_path IF NOT EXISTS
FOR (n:File)
ON (n.project_id, n.file_path);
```

## Import policy

Raw Joern `neo4jcsv` is an extraction format, not the canonical contract.

Import pipeline:

```text
raw Joern CSV
 -> normalize labels/properties/edge types
 -> validate project_id
 -> validate endpoints
 -> canonical CSV
 -> Neo4j
```

Before creating relationships:
- both endpoint UIDs must exist;
- both endpoints must have the same project_id;
- relation project_id must equal the endpoint project_id.

For large initial graph imports use the bulk importer.
For bounded updates/enrichment use Cypher transactions.

## Standard read patterns

### Find a method

```cypher
MATCH (m:Method {project_id: $project_id})
WHERE m.full_name = $full_name OR m.name = $name
RETURN m
LIMIT 50;
```

### Direct callers

```cypher
MATCH (caller:Method {project_id: $project_id})
      -[:CALLS {project_id: $project_id}]->
      (target:Method {project_id: $project_id})
WHERE target.full_name = $full_name
RETURN caller, target;
```

### Direct callees

```cypher
MATCH (source:Method {project_id: $project_id})
      -[:CALLS {project_id: $project_id}]->
      (callee:Method {project_id: $project_id})
WHERE source.full_name = $full_name
RETURN source, callee;
```

### Bounded call path

```cypher
MATCH p =
  (source:Method {project_id: $project_id})
  -[:CALLS*1..8]->
  (target:Method {project_id: $project_id})
WHERE source.full_name = $source_full_name
  AND target.full_name = $target_full_name
  AND all(n IN nodes(p) WHERE n.project_id = $project_id)
RETURN p
LIMIT 20;
```

Always use a bounded traversal unless an unbounded traversal is explicitly justified.

### File -> methods

```cypher
MATCH (f:File {project_id: $project_id})
      -[:CONTAINS|DECLARES*1..3]->
      (m:Method {project_id: $project_id})
WHERE f.file_path = $file_path
RETURN m;
```

## Path reporting

When returning a path to the agent include:

- `project_id`
- node kind
- name/full_name
- file path
- line range
- relationship type
- relationship source
- graph schema version

Prefer paths that can be mapped back to source code.

## Agent enrichment

Allowed examples:

```text
(:Finding)
(:EntryPoint)
(:Sink)
(:Source)
(:Precondition)
(:PoC)
(:Technique)
```

These are analytical nodes, not replacements for code nodes.

They MUST include `project_id`.

Recommended pattern:

```cypher
MERGE (f:Finding {
  project_id: $project_id,
  finding_id: $finding_id
})
SET f.source = "agent",
    f.schema_version = "1";
```

When linking an analytical node to code:

```cypher
MATCH (f:Finding {project_id: $project_id, finding_id: $finding_id}),
      (m:Method {project_id: $project_id, uid: $method_uid})
MERGE (f)-[:EVIDENCED_BY {project_id: $project_id, source: "agent"}]->(m);
```

Do not create agent edges that masquerade as Joern facts.

## Rebuild policy

For source re-indexing, default to project-level replacement:

```cypher
MATCH (n {project_id: $project_id})
DETACH DELETE n;
```

then import the new normalized graph.

If analytical findings must survive a rebuild, store them separately or rebind them after import using stable evidence keys:
- repository revision;
- file path;
- symbol full name;
- source line range;
- content hash.

Do not assume Joern internal node IDs remain stable across rebuilds.

## Validation queries

### Cross-project relationships

This must return zero:

```cypher
MATCH (a)-[r]->(b)
WHERE a.project_id <> b.project_id
   OR r.project_id <> a.project_id
   OR r.project_id <> b.project_id
RETURN count(r) AS invalid_relationships;
```

### Missing UID

```cypher
MATCH (n:CodeNode)
WHERE n.uid IS NULL OR n.project_id IS NULL OR n.node_id IS NULL
RETURN count(n) AS invalid_nodes;
```

### Duplicate UID

```cypher
MATCH (n:CodeNode)
WITH n.uid AS uid, count(*) AS c
WHERE c > 1
RETURN uid, c;
```

### Schema versions

```cypher
MATCH (n:CodeNode)
RETURN n.schema_version, count(*)
ORDER BY n.schema_version;
```

## Safety/performance rules

1. Default to read-only Cypher.
2. Never run `DETACH DELETE` without an explicit `project_id`.
3. Never run graph-wide updates merely to fix one project.
4. Bound variable-length traversals.
5. Return only fields needed by the agent.
6. Use parameters.
7. Prefer indexes/constraints on merge keys.
8. Before a large write, count the intended target set.
9. After import, run validation queries.
10. Never infer identity across applications.

## Output contract

Return results in a project-scoped structure:

```json
{
  "project_id": "saleor",
  "query_type": "call_path",
  "paths": [],
  "warnings": []
}
```

Explicitly distinguish:
- Joern-derived code facts;
- canonicalized facts;
- agent-created inference.
