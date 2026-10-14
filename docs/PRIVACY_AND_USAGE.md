# Privacy, Usage, and Retention Notes

AI-NIDS is an educational/research prototype. This document describes the current local
Streamlit dashboard behavior; it is not legal advice and is not a substitute for a policy
review before any public or commercial deployment.

## Data handled

The dashboard reads model-generated records from `artifacts/alerts.jsonl`. The expected fields
include timestamp, source file, source row, predicted class, confidence, and class probabilities.
The dashboard does not provide forms for collecting names, email addresses, telephone numbers,
payment information, or other user profile data.

## Local processing and third parties

The app does not intentionally send alert records to an external analytics or advertising service.
Streamlit's own runtime and any hosting provider may process technical/session data. Review the
provider's current terms and configuration before hosting the app publicly.

## Retention and deletion

The dashboard does not automatically delete or rotate alert records. Operators must manage
`artifacts/alerts.jsonl` and related model artifacts according to their own retention needs.
For a local reset, stop the app and remove the generated alert log only if the records are no
longer required. Do not delete trained model artifacts unless you can regenerate them.

## Security and limitations

Predictions are not confirmed incidents, confidence is not accuracy, and the tool does not
guarantee that traffic predicted as Normal is safe. Do not expose the API or dashboard to an
untrusted network without an appropriate deployment security review, authentication, access
controls, rate limiting, and monitoring.

## Cookies, consent, age, and commercial policies

This research dashboard has no account creation, marketing email, subscription, checkout, refund,
or customer-contact workflow. Therefore refund policy, unsubscribe link, business phone/email
links, customer form consent, and age-consent flows are not implemented. No custom cookie banner
is added because the project does not configure advertising/analytics cookies; hosting and
Streamlit session behavior must still be reviewed for a real deployment. If the project later
collects personal data or offers services to customers, create a reviewed privacy notice, terms,
cookie consent flow where required, data-subject request process, and any age-specific protections
before launch.
