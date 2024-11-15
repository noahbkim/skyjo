#ifndef TEST_H
#define TEST_H

// MARK: Framework

#define TESTS_BEGIN() \
    int main()        \
    {                 \
        const char *TEST_NAME = NULL;
#define TESTS_END() }
#define TEST(NAME) TEST_NAME = #NAME;

#define STRING_INNER(VALUE) #VALUE
#define STRING(VALUE) STRING_INNER(VALUE)
#define WHERE() __FILE__ ":" STRING(__LINE__)

#define ASSERT(CONDITION)                                   \
    do                                                      \
    {                                                       \
        if (!(CONDITION))                                   \
        {                                                   \
            fprintf(stderr,                                 \
                    "Failed %s in " WHERE() ": " #CONDITION \
                                            "\n",           \
                    TEST_NAME);                             \
        }                                                   \
    } while (0)

// MARK: Hand

// clang-format off

#define C -CARD_SAFE
#define H +CARD_SAFE

#define _decode_finger_present(CODE) (CODE != C)

#define _decode_finger_state(CODE) \
    ((CODE < CARD_MIN) ? CARD_CLEARED : \
     (CODE > CARD_MAX) ? CARD_HIDDEN :  \
                         CARD_REVEALED)

#define _decode_finger_card(CODE) \
    ((CODE < CARD_MIN) ? 0 : \
     (CODE > CARD_MAX) ? CODE - CARD_SAFE : \
                         CODE)

#define finger_matches(FINGER, CODE) \
    (FINGER.state == _decode_finger_state(CODE) && \
     FINGER.card == _decode_finger_card(CODE))

#define hand_matches(HAND, C00, C01, C02, C03, C10, C11, C12, C13, C20, C21, C22, C23) \
    (finger_matches(HAND.fingers[0], C00) \
     && finger_matches(HAND.fingers[1], C10) \
     && finger_matches(HAND.fingers[2], C20) \
     && finger_matches(HAND.fingers[3], C01) \
     && finger_matches(HAND.fingers[4], C11) \
     && finger_matches(HAND.fingers[5], C21) \
     && finger_matches(HAND.fingers[6], C02) \
     && finger_matches(HAND.fingers[7], C12) \
     && finger_matches(HAND.fingers[8], C22) \
     && finger_matches(HAND.fingers[9], C03) \
     && finger_matches(HAND.fingers[10], C13) \
     && finger_matches(HAND.fingers[11], C23))

#define finger_assign(FINGER, CODE) \
    FINGER.state = _decode_finger_state(CODE); \
    FINGER.card = _decode_finger_card(CODE);

#define hand_assign(HAND, C00, C01, C02, C03, C10, C11, C12, C13, C20, C21, C22, C23) \
    finger_assign(HAND.fingers[0], C00); \
    finger_assign(HAND.fingers[1], C10); \
    finger_assign(HAND.fingers[2], C20); \
    finger_assign(HAND.fingers[3], C01); \
    finger_assign(HAND.fingers[4], C11); \
    finger_assign(HAND.fingers[5], C21); \
    finger_assign(HAND.fingers[6], C02); \
    finger_assign(HAND.fingers[7], C12); \
    finger_assign(HAND.fingers[8], C22); \
    finger_assign(HAND.fingers[9], C03); \
    finger_assign(HAND.fingers[10], C13); \
    finger_assign(HAND.fingers[11], C23); \
    HAND.columns = (_decode_finger_present(C00) \
        + _decode_finger_present(C01) \
        + _decode_finger_present(C02) \
        + _decode_finger_present(C03))

// clang-format on

#endif // TEST_H
