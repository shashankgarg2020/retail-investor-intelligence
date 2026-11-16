import pytest

from src.processing import quality
from src.processing.entity_resolution import IRRELEVANT, POSSIBLY, RELEVANT, resolve

CASES = [
    # Clear investor discussion.
    ("VEON", "VEON earnings beat, revenue up 12% and net debt down", RELEVANT),
    ("VEON", "Bought more $VEON today", RELEVANT),
    ("TTWO", "Take-Two Interactive raised bookings guidance for the fiscal year", RELEVANT),
    ("STB", "Secure Trust Bank shares fell after the FCA motor finance update", RELEVANT),
    ("STB", "Is STB.L undervalued at this price?", RELEVANT),
    # Product chatter is not investor discussion.
    ("TTWO", "Just finished the GTA 6 trailer breakdown, the map looks huge", IRRELEVANT),
    ("TTWO", "Rockstar should add more heists", IRRELEVANT),
    # Ambiguous ticker resolved by disambiguators.
    ("STB", "My STB keeps rebooting, is the set-top box firmware broken?", IRRELEVANT),
    ("STB", "Sacombank STB shares hit a new high on HOSE", IRRELEVANT),
    ("STB", "STB looks interesting", IRRELEVANT),
    # Ambiguous term with full support.
    ("STB", "STB is a small UK bank, shares look cheap versus other lenders after earnings", RELEVANT),
    # Needs a second opinion.
    ("TTWO", "If GTA VI slips again the stock will get hammered", POSSIBLY),
    ("VEON", "Kyivstar valuation after the listing looks stretched", POSSIBLY),
    # Nothing to go on.
    ("VEON", "Great weather today", IRRELEVANT),
]


@pytest.mark.parametrize("ticker,text,expected", CASES)
def test_rules(ticker, text, expected):
    result = resolve(ticker, text)
    assert result.label == expected, result.reason
    assert result.method == "rules-v2" and result.reason


def test_company_separation():
    text = "VEON earnings beat expectations"
    assert resolve("VEON", text).label == RELEVANT
    assert resolve("TTWO", text).label == IRRELEVANT
    assert resolve("STB", text).label == IRRELEVANT


def test_comment_inherits_context_from_parent():
    parent = "Take-Two Interactive Q2 earnings call reaction"
    assert resolve("TTWO", "Guidance was weak and margins are shrinking", parent).label == RELEVANT
    assert resolve("TTWO", "first!", parent).label == POSSIBLY
    assert resolve("TTWO", "Guidance was weak", "Top 10 open world games").label == IRRELEVANT


def test_cashtag_is_not_matched_inside_other_words():
    assert resolve("VEON", "the veonix project").label == IRRELEVANT


def test_exact_case_identifier():
    # "VEON" the company is written in capitals; "veon"/"Veon" alone is usually a name or a character.
    assert resolve("VEON", "VEON signs a new tower deal").label == RELEVANT
    assert resolve("VEON", "A very sweet shiny veon").label == IRRELEVANT
    assert resolve("VEON", "Le'Veon Bell was overrated, the Steelers moved on").label == IRRELEVANT


def test_strong_match_with_disambiguator_goes_to_llm():
    assert resolve("VEON", "VEON — I'm Already Here #MelodicTechno").label == POSSIBLY
    # Investor vocabulary still wins: the identifier plus finance terms is unambiguous.
    assert resolve("VEON", "VEON earnings call #MelodicTechno playlist on").label == RELEVANT


def test_product_news_goes_to_llm_but_chatter_does_not():
    assert resolve("TTWO", "Red Dead Redemption 2 has now sold 89 million copies").label == POSSIBLY
    assert resolve("TTWO", "GTA6 delayed until March").label == POSSIBLY
    assert resolve("TTWO", "Rockstar should add more heists").label == IRRELEVANT
    assert resolve("STB", "W or L trade for two mutated cannons in STB #fortnite").label == IRRELEVANT


def test_relevance_ignores_sentiment():
    assert resolve("VEON", "VEON is a terrible stock").label == resolve("VEON", "VEON is a great stock").label


# ---------------------------------------------------------------------- quality

def test_quality_flags():
    assert quality.assess_text("VEON reported solid results and the debt picture is improving")[0] == quality.OK
    assert quality.assess_text("$VEON 🚀")[0] == quality.LOW_QUALITY
    assert quality.assess_text("Watch $AAPL $TSLA $NVDA $AMD $VEON $TTWO $MSFT movers today right now")[0] == quality.LOW_QUALITY
    assert quality.assess_text("VEON analysis inside, join our Telegram for free signals every day")[0] == quality.PROMOTIONAL
    assert quality.assess_text("Read more https://a.co/1 https://a.co/2 https://a.co/3 about VEON results")[0] == quality.PROMOTIONAL


def test_repeated_templates_flag_content_not_people():
    template = "VEON week ahead: options market expects a {}% move by Friday, key levels inside"
    posts = [{"id": i, "author_id": "feed", "text": template.format(i)} for i in range(1, 4)]
    posts.append({"id": 9, "author_id": "human", "text": template.format(5)})
    posts.append({"id": 10, "author_id": "feed", "text": "A completely different thought about VEON and its debt"})
    assert quality.repeated_templates(posts) == {1, 2, 3}


def test_language_detection_is_cautious():
    assert quality.detect_language("VEON reported strong quarterly earnings and raised guidance") == "en"
    assert quality.detect_language("Sacombank công bố kết quả kinh doanh quý ba với lợi nhuận tăng mạnh") == "vi"
    assert quality.detect_language("$VEON") == "und"
    assert quality.detect_language("anything", platform_hint="de") == "de"
