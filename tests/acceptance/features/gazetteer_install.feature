Feature: Gazetteer installation from the CLI

  Scenario: Installing a gazetteer from a local config
    Given a gazetteer config pointing at the local Andorra fixture
    When I run the geoparser command "install" with that config
    Then the exit code is 0
    And the installed gazetteer finds "Andorra la Vella"

  Scenario: Installing an unknown gazetteer fails cleanly
    When I run the geoparser command "install" with "nonexistent"
    Then the exit code is 2
    And the error output names "nonexistent" without a traceback
