from benchmarker import Benchmarker

## specify number of loop
with Benchmarker(2000*2000, width=20) as bench:
    s1, s2, s3, s4, s5 = "Haruhi", "Mikuru", "Yuki", "Itsuki", "Kyon"

    @bench(None)                ## empty loop
    def _(bm):
        for i in bm:
            pass

    @bench("join")
    def _(bm):
        for i in bm:
            sos = ''.join((s1, s2, s3, s4, s5))

    @bench("concat")
    def _(bm):
        for i in bm:
            sos = s1 + s2 + s3 + s4 + s5

    @bench("format")
    def _(bm):
        for i in bm:
            sos = '%s%s%s%s%s' % (s1, s2, s3, s4, s5)