from typing import Any

from neo4j import Driver

CYPHER_CHECK_QUERY = """
MATCH (i:UserIntent {id: $intent_id})

// 1. Предки переданных активов (если есть)
OPTIONAL MATCH (a:Asset)-[:IS_A*1..3]->(from_asset:AssetClass)
WHERE a.id IN $asset_ids

// 2. Сами переданные классы И ВСЕ ИХ ПРЕДКИ (CLASS_EQUITY -> CLASS_SECURITY)
OPTIONAL MATCH (d:AssetClass)-[:IS_A*0..3]->(direct_cls:AssetClass)
WHERE d.id IN $class_ids

WITH i, [c IN (collect(DISTINCT from_asset) + collect(DISTINCT direct_cls))
 WHERE c IS NOT NULL] AS target_classes

// 3. Поиск политики
MATCH (p:SecurityPolicy)-[:RESTRICTS]->(i)
WHERE
  // Случай А: абстрактный запрос без инструментов и классов вообще
  (size(target_classes) = 0 AND p.id = 'POL_RB_SECURITIES_231Z')

  // Случай Б: совет по запрещенным классам
  OR ANY(cls IN target_classes WHERE (p)-[:APPLIES_TO_CLASS]->(cls))

  // Случай В: сравнение токенов с защищенным вкладом
  OR (p.id = 'POL_RB_DEPOSIT_COMPARE' AND EXISTS {
        MATCH (dep:Asset)-[:IS_A*1..3]->(dep_cls:AssetClass {state_guaranteed: true})
        WHERE dep.id IN $asset_ids
        AND ANY(c IN target_classes WHERE c.id = 'CLASS_TOKEN')
  })

MATCH (p)-[:RESULTS_IN]->(v:Verdict {status: "DENY"})

RETURN
    v.status AS verdict,
    p.id AS policy,
    p.legal_reference AS law,
    p.reason AS reason,
    [i.id, p.id, v.status] AS trace
LIMIT 1;
"""


class RuleEngine:
    def __init__(self, neo4j_driver: Driver) -> None:
        self.driver = neo4j_driver

    def evaluate(
        self, intent_id: str, asset_ids: list[str], class_ids: list[str]
    ) -> dict[str, Any]:
        with self.driver.session() as session:
            result = session.run(
                CYPHER_CHECK_QUERY,
                intent_id=intent_id,
                asset_ids=asset_ids,
                class_ids=class_ids,
            ).single()

            if result:
                return {
                    "verdict": result["verdict"],
                    "policy": result["policy"],
                    "law": result["law"],
                    "reason": result["reason"],
                    "trace": result["trace"],
                }

            return {
                "verdict": "ALLOW",
                "policy": None,
                "law": None,
                "reason": None,
                "trace": [],
            }
