Feature: Evaluate pilot evidence

  Scenario: A correctly resolved Andorran capital scores perfectly
    Given the pilot observed "Andorra la Vella is the capital of Andorra."
    And the gold place is "Andorra la Vella" with identifier "3041563"
    And the predicted place is "Andorra la Vella" with identifier "3041563"
    When I build the pilot report
    Then the report recognition F1 is 1.0
    And the report resolution accuracy is 1.0
    And the report preserves the span text "Andorra la Vella"

  Scenario: A missed place is not covered up by the same offsets elsewhere
    Given two pilot documents whose gold places share the same offsets
    And only the first document is recognized and resolved
    When I build the pilot report for both documents
    Then the aggregate recognition recall is 0.5
    And the aggregate resolution accuracy is 0.5
    And the aggregate counts 2 gold annotations

  Scenario: Predictions stay with their document when the database reorders them
    Given two pilot documents read back in reverse order
    When I collect the pilot predictions by document ID
    Then each document keeps its own predicted span
