from typing import Any, Dict, List, Optional, TypedDict


class QueryQualityScore(TypedDict):
    completeness: float
    clarity: float
    overall: float
    issues: List[str]


class QueryEvaluator:
    def evaluate(self, user_query: str, parsed_question: Dict[str, Any], search_result: List[Dict[str, Any]]) -> QueryQualityScore:
        issues: list[str] = []
        completeness = 1.0
        clarity = 1.0

        if not user_query or not user_query.strip():
            issues.append("用户查询为空")
            completeness = 0.0
            clarity = 0.0

        event = parsed_question.get("event") if parsed_question else None
        if not event or (isinstance(event, str) and event.lower() == "null"):
            issues.append("缺少事件类型描述")
            completeness -= 0.3

        result_count = len(search_result)
        if result_count == 0:
            issues.append("检索结果为空")
            completeness = 0.3
        elif result_count < 3:
            issues.append(f"检索结果过少({result_count}条)，可能查询条件过于严格")
            completeness = 0.5
        elif result_count > 100:
            issues.append(f"检索结果过多({result_count}条)，可能查询条件过于宽松")
            completeness = 0.7

        if completeness < 0.5:
            issues.append("查询完整性不足，需要优化")

        overall = (completeness + clarity) / 2.0

        return QueryQualityScore(
            completeness=round(completeness, 3),
            clarity=round(clarity, 3),
            overall=round(overall, 3),
            issues=issues,
        )

    def is_satisfactory(self, quality_score: QueryQualityScore) -> bool:
        # Only the "overall" threshold matters for the satisfication check.
        # Issues are informational and should not gate the result because
        # evaluate() appends them for nearly every query (result count,
        # missing event, etc.), making len(issues)==0 nearly impossible.
        return quality_score.get("overall", 0.0) >= 0.6


def create_query_evaluator() -> QueryEvaluator:
    return QueryEvaluator()


if __name__ == "__main__":
    evaluator = create_query_evaluator()

    result1 = evaluator.evaluate("车进入镜头", {"event": "进入", "color": None}, [{"event_id": 1}])
    print("good query:", result1)

    result2 = evaluator.evaluate("进入", {"event": None}, [])
    print("bad query:", result2)

    print("is satisfactory:", evaluator.is_satisfactory(result1))