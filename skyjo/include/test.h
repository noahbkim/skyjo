#ifndef TEST_H
#define TEST_H

#define TESTS_BEGIN() \
    int main()        \
    {                 \
        const char *TEST_NAME = NULL;
#define TESTS_END() }
#define TEST(NAME) TEST_NAME = #NAME;
#define STRING_INNER(VALUE) #VALUE
#define STRING(VALUE) STRING_INNER(VALUE)
#define WHERE() __FILE__ ":" STRING(__LINE__)
#define ASSERT(CONDITION)                                     \
    do                                                        \
    {                                                         \
        if (!(CONDITION))                                     \
        {                                                     \
            fprintf(                                          \
                stderr,                                       \
                "Failed %s in " WHERE() ": " #CONDITION "\n", \
                TEST_NAME);                                   \
        }                                                     \
    } while (0)

#endif // TEST_H
