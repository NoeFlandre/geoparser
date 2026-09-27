Feature: Annotator session

  Scenario: Uploading a document and annotating a toponym
    Given the annotator app is running on a test client
    When I upload "Andorra la Vella is small." for the Andorra gazetteer
    And I mark characters 0 to 16 as a toponym
    And I select candidate "3041563" for that toponym
    Then the exported session places characters 0 to 16 at "3041563"
