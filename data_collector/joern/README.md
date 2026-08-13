# CodeGraph Agent Stack

Централизованный контур для анализа разных приложений через Joern и Neo4j.

## 1. Архитектура

```text
                  +----------------------+
                  |       AI Agent       |
                  +----------+-----------+
                             |
              +--------------+----------------+
              |                               |
              v                               v
      +---------------+               +---------------+
      | Joern Server  |               |    Neo4j      |
      | :8080 HTTP    |               | :7687 Bolt    |
      +-------+-------+               +-------+-------+
              |                               ^
              | import/query                  |
              v                               | import/query
      /projects/<project_id>/src              |
              |                               |
              v                               |
      Joern CPG / workspace                   |
              |                               |
              v                               |
      /exports/<project_id>/raw/neo4jcsv -----+
              |
              v
      /exports/<project_id>/canonical/
```

Принцип: один инфраструктурный Joern, один Neo4j, много изолированных проектов.

Межпроектные связи запрещены. У каждого проекта есть стабильный `project_id`.

## 2. Каноническая схема

Joern CPG используется как исходная модель. Для AI-агента фиксируется небольшой стабильный словарь.

### Canonical node labels

- `Project`
- `File`
- `Namespace`
- `Type`
- `Method`
- `Parameter`
- `Local`
- `Call`
- `Identifier`
- `Literal`
- `ControlStructure`
- `Member`
- `Return`
- `Unknown`

Не надо переименовывать узлы в зависимости от языка:
`PythonFunction`, `JavaMethod`, `GoFunction` и т.п. запрещены.
Все функции/методы -> `Method`.

### Canonical relationship types

- `CONTAINS`
- `DECLARES`
- `AST_CHILD`
- `CALLS`
- `ARGUMENT`
- `RETURNS`
- `CFG_NEXT`
- `CDG_DEPENDS_ON`
- `DDG_DEPENDS_ON`
- `REFERS_TO`
- `BINDS`
- `INHERITS_FROM`
- `IMPLEMENTS`
- `HAS_TYPE`

Названия типов ребер всегда в `UPPER_SNAKE_CASE`.

### Mandatory node properties

Каждый узел должен иметь:

- `project_id`
- `node_id`
- `kind`
- `name` (если применимо)
- `full_name` (если Joern предоставляет)
- `file_path` (если применимо)
- `line_start`
- `line_end`
- `language`
- `source = "joern"`
- `schema_version = "1"`

`node_id` должен быть уникален только внутри проекта.
Глобальный ключ: `<project_id>:<node_id>`.

### Mandatory relationship properties

Каждое ребро должно иметь:

- `project_id`
- `edge_id`
- `source_node_id`
- `target_node_id`
- `kind`
- `source = "joern"`
- `schema_version = "1"`

Инвариант:

```text
source.project_id == relationship.project_id == target.project_id
```

Нельзя создавать ребро, если project_id концов различается.

## 3. Развертывание

### Каталоги

```bash
sudo mkdir -p /opt/codegraph/{projects,exports,neo4j/data,neo4j/logs,neo4j/import}
sudo chown -R "$USER":"$USER" /opt/codegraph
cd /opt/codegraph
```

Скопируйте в `/opt/codegraph` файлы этого пакета.

### .env

```bash
cp .env.example .env
nano .env
```

### Запуск

```bash
docker compose up -d
docker compose ps
```

Joern HTTP API:
`http://127.0.0.1:8080`

Neo4j Browser:
`http://127.0.0.1:7474`

Bolt:
`bolt://127.0.0.1:7687`

Joern server поддерживает HTTP запросы к interpreter. Для production не публикуйте 8080 наружу.

## 4. Подключение нового приложения

Проект идентифицируется коротким slug:

```text
saleor
grafana
my-service
```

Исходники:

```bash
mkdir -p projects/saleor
git clone https://github.com/saleor/saleor.git projects/saleor/src
```

Структура:

```text
projects/
  saleor/
    src/
exports/
  saleor/
    raw/
    canonical/
```

### Импорт в Joern

Через server API:

```bash
curl -s \
  -u "$JOERN_USER:$JOERN_PASSWORD" \
  http://127.0.0.1:8080/query-sync \
  -H 'Content-Type: application/json' \
  -d '{"query":"importCode(\"/projects/saleor/src\", \"saleor\")"}'
```

После импорта Joern хранит CPG в своем workspace.

### Запрос к CPG

```bash
curl -s \
  -u "$JOERN_USER:$JOERN_PASSWORD" \
  http://127.0.0.1:8080/query-sync \
  -H 'Content-Type: application/json' \
  -d '{"query":"workspace.setActiveProject(\"saleor\"); cpg.method.name.l"}'
```

Агент всегда должен сначала активировать нужный проект.

## 5. Экспорт Neo4j CSV

Joern умеет экспортировать полный граф в `neo4jcsv`.

Используйте централизованный helper:

```bash
./scripts/export-project.sh saleor
```

Результат:

```text
exports/saleor/raw/neo4jcsv/
```

Важно: raw export считается техническим артефактом Joern.
AI-агент должен работать преимущественно с canonical graph в Neo4j.

## 6. Canonical normalization

Рекомендуемый pipeline:

```text
Joern CPG
  -> raw neo4jcsv
  -> normalizer
  -> canonical nodes.csv / relationships.csv
  -> Neo4j
```

Normalizer выполняет:

1. добавление `project_id`;
2. преобразование Joern labels к canonical labels;
3. преобразование edge types к canonical relationship types;
4. сохранение исходного типа в `joern_label` / `joern_edge_type`;
5. построение глобального `uid = project_id + ":" + node_id`;
6. проверку отсутствия cross-project edges;
7. запись `schema_version=1`.

Рекомендуется никогда не удалять исходную Joern-семантику:
canonical label служит стабильным API, а `joern_label` сохраняет точность CPG.

## 7. Загрузка в Neo4j

Для небольших/инкрементальных наборов допустим `LOAD CSV`.
Для полного массового импорта предпочтителен `neo4j-admin database import full`.

Минимальные constraints:

```cypher
CREATE CONSTRAINT code_node_uid IF NOT EXISTS
FOR (n:CodeNode)
REQUIRE n.uid IS UNIQUE;

CREATE INDEX code_node_project IF NOT EXISTS
FOR (n:CodeNode)
ON (n.project_id);

CREATE INDEX method_lookup IF NOT EXISTS
FOR (n:Method)
ON (n.project_id, n.full_name);

CREATE INDEX file_lookup IF NOT EXISTS
FOR (n:File)
ON (n.project_id, n.file_path);
```

Рекомендуется давать каждому каноническому узлу дополнительный общий label `CodeNode`:

```text
(:CodeNode:Method {...})
(:CodeNode:File {...})
(:CodeNode:Call {...})
```

Это позволяет иметь единственный global UID constraint.

## 8. Изоляция проектов

В Neo4j Community одна стандартная database на DBMS, поэтому для этого стека используется общий database + `project_id`.

Обязательные правила:

- каждый MATCH агента содержит `project_id`;
- каждый CREATE/MERGE содержит `project_id`;
- никакие эвристики не связывают сущности разных проектов;
- одинаковый `full_name` в двух приложениях не означает одну сущность;
- очистка делается только по project_id:

```cypher
MATCH (n {project_id: $project_id})
DETACH DELETE n;
```

Если в будущем появится Neo4j Enterprise, можно выделить отдельную database на приложение, сохранив ту же canonical schema.

## 9. Жизненный цикл проекта

```text
REGISTER
  -> IMPORT_JOERN
  -> VALIDATE_CPG
  -> EXPORT_RAW
  -> NORMALIZE
  -> LOAD_NEO4J
  -> VALIDATE_GRAPH
  -> READY
```

Повторный анализ:

```text
SOURCE_CHANGED
  -> DROP/REBUILD Joern project
  -> EXPORT_RAW
  -> DELETE Neo4j nodes by project_id
  -> NORMALIZE
  -> LOAD_NEO4J
  -> VALIDATE_GRAPH
```

На первом этапе лучше делать deterministic full rebuild, а не incremental merge.

## 10. Что должен знать агент

Joern:
- источник фактов о коде;
- AST/CFG/call/data-flow traversals;
- построение CPG;
- raw export.

Neo4j:
- долговременная рабочая графовая модель;
- поиск путей;
- объединение разных типов связей внутри одного приложения;
- добавление агентом собственных сущностей и связей поверх CPG.

Не смешивать обязанности:
Joern не использовать как долговременную knowledge base.
Neo4j не использовать как замену Joern semantic/data-flow engine.
