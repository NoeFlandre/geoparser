Feature: Evaluating resolution distance

  Scenario: A prediction 100 km off counts at 161 km but not at 50 km
    Given a gold place and a prediction 100 km away
    Then accuracy at 161 km is 1.0
    And accuracy at 50 km is 0.0
