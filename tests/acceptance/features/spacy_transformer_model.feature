Feature: Choose a spaCy model for recognition

  The transformer model needs a plugin that spaCy does not install itself.
  When that plugin is missing, the recognizer should explain the fix.

  Scenario: A caller builds a recognizer on a transformer model without the plugin
    Given the spaCy model "en_core_web_trf" needs the missing transformer plugin
    When I build a recognizer on "en_core_web_trf"
    Then I am told to install "spacy-curated-transformers"
    And I am still shown the original spaCy error
