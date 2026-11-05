// Example Cypher queries for the AML graph (model: src/qlmaml/graphdb.py).
// Paste into Neo4j Browser (http://localhost:7474). Times are UTC.

// 1. Overview: transactions and laundering per split.
MATCH ()-[t:TRANSFER]->()
RETURN t.split AS split, count(*) AS n, sum(t.is_laundering) AS laundering
ORDER BY split;

// 2. Ground-truth motif counts from *_Patterns.txt.
MATCH (p:LaunderingAttempt)
RETURN p.pattern AS pattern, count(*) AS attempts, avg(p.n_tx) AS avg_tx
ORDER BY attempts DESC;

// 3. Look at one laundering attempt as a picture (Browser graph view).
MATCH (p:LaunderingAttempt {pattern: 'CYCLE'})
WITH p LIMIT 1
MATCH (a)-[t:TRANSFER {attempt_id: toInteger(p.attempt_id)}]->(b)  // node id is a string after neo4j-admin import
RETURN a, t, b;

// 4. Fan-out: accounts that pay >= 5 distinct receivers within 24 h (train split).
MATCH (a:Account)-[t:TRANSFER {split: 'train'}]->(b:Account)
WITH a, t.ts AS ts, b
ORDER BY ts
WITH a, collect({ts: ts, b: b.account_id}) AS out
UNWIND out AS e
WITH a, e, [x IN out WHERE x.ts >= e.ts AND x.ts <= e.ts + duration('P1D') | x.b] AS window
WITH a, e.ts AS start, size(apoc.coll.toSet(window)) AS k
WHERE k >= 5
RETURN a.account_id, start, k ORDER BY k DESC LIMIT 25;

// 5. Temporal 3-cycles: a -> b -> c -> a, each hop later than the previous, within 3 days.
MATCH (a:Account)-[t1:TRANSFER]->(b:Account)-[t2:TRANSFER]->(c:Account)-[t3:TRANSFER]->(a)
WHERE a <> b AND b <> c AND a <> c
  AND t1.ts < t2.ts < t3.ts AND t3.ts <= t1.ts + duration('P3D')
RETURN a.account_id, b.account_id, c.account_id, t1.is_laundering + t2.is_laundering + t3.is_laundering AS n_laund
LIMIT 25;

// 6. Simple-graph view (one edge per account pair): heaviest pairs in train.
MATCH (a)-[s:SENDS_TO {split: 'train'}]->(b)
RETURN a.account_id, b.account_id, s.n_tx, s.total_paid, s.n_laundering
ORDER BY s.n_tx DESC LIMIT 25;

// 7. Ego-net around one laundering transaction (1 hop, 24 h before it).
//    Python: GraphDB().ego_subgraph(edge_id, hops=1, max_nodes=12) does this with a node cap.
MATCH (a)-[t:TRANSFER {is_laundering: 1}]->(b)
WITH a, b, t LIMIT 1
MATCH (x:Account)-[r:TRANSFER]-(y:Account)
WHERE (x = a OR x = b) AND t.ts - duration('P1D') <= r.ts <= t.ts
RETURN x, r, y;

// 8. GDS: project the train simple graph and run PageRank (needs the graph-data-science plugin).
// CALL gds.graph.project.cypher('train',
//   'MATCH (a:Account) RETURN id(a) AS id',
//   'MATCH (a)-[s:SENDS_TO {split: "train"}]->(b) RETURN id(a) AS source, id(b) AS target, s.n_tx AS weight');
// CALL gds.pageRank.stream('train', {relationshipWeightProperty: 'weight'})
// YIELD nodeId, score RETURN gds.util.asNode(nodeId).account_id AS account, score ORDER BY score DESC LIMIT 20;
