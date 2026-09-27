Feature: Project persistence

  Scenario: Results survive reopening a project
    Given I parsed "Andorra la Vella is small." in project "p1" with the manual pipeline
    When I open project "p1" again
    Then the document still has the place "Andorra la Vella" resolved to "3041563"
