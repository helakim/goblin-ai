# *********************************************************************
# @Project    text_summary
# @FILE       kyung_tae_kim (hela.kim)
# @Copyright: hela
# @Created:   29/09/2019
#
#            7''  Q..\
#         _7         (_
#       _7  _/    _q.  /
#     _7 . ___  /VVvv-'_                                            .
#    7/ / /~- \_\\      '-._     .-'                      /       //
#   ./ ( /-~-/||'=.__  '::. '-~'' {             ___   /  //     ./{
#  V   V-~-~| ||   __''_   ':::.   ''~-~.___.-'' _/  // / {_   /  {  /
#   VV/-~-~-|/ \ .'__'. '.    '::                     _ _ _        ''.
#   / /~~~~||VVV/ /  \ )  \        _ __ ___   ___ ___(_) | | __ _   .::'
#  / (~-~-~\\.-' /    \'   \::::. | '_ ` _ \ / _ \_  / | | |/ _` | :::'
# /..\    /..\__/      '     '::: | | | | | | (_) / /| | | | (_| | ::
# vVVv    vVVv                 ': |_| |_| |_|\___/___|_|_|_|\__,_| ''
#
# *********************************************************************
# Todo Will be convert to class
# References: https://docs.google.com/spreadsheets/d/1OGAjUvalBuX-oZvZ_-9tEfYD2gQe7hTGsgUpiiBSXI8/edit#gid=0
MECAB_KO_USEFUL_TAGS = list(['NNP', 'NNB', 'NNBC', 'VV', 'XR', 'IC', 'JKS', 'NNG'])
""" Korean(mecab) Pos tags chart
1. NNG	일반 명사
2. NNP	고유 명사
3. NNB	의존 명사
4. NNBC	단위를 나타내는 명사
5. NR	수사
6. NP 	대명사
7. VV	동사
8. VA	형용사
9 .VX	보조 용언
10. VCP	긍정 지정사
11. VCN	부정 지정사
12. MM	관형사
13. MAG	일반 부사
14. MAJ	접속 부사
15. IC	감탄사
16. JKS	주격 조사
17. JKC	보격 조사
18. JKG	관형격 조사
19. JKO	목적격 조사
20. JKB	부사격 조사
21. JKV	호격 조사
22. JKQ	인용격 조사
23. JC	접속 조사
24. JX	보조사
25. EP	선어말어미
26. EF	종결 어미
27. EC	연결 어미
28. ETN	명사형 전성 어미
29. ETM	관형형 전성 어미
30. XPN	체언 접두사
31. XSN	명사파생 접미사
32. XSV	동사 파생 접미사
33. XSA	형용사 파생 접미사
34. XR	어근
35. SF	마침표, 물음표, 느낌표
36. SE	줄임표 …
37. SSO	여는 괄호 (, [
38. SSC	닫는 괄호 ), ]
39. SC	구분자 , · / :
40. SY	기타 기호
41. SH	한자
42. SL	외국어
43. SN	숫자
"""
TWITTER_KO_USEFUL_TAGS = list(['Noun', 'Verb', 'Adjective', 'KoreanParticle', 'Number', 'Adverb', 'Modifier'])
""" Twitter korean text
Noun          명사
Verb          동사
Adjective     형용사
Determiner    관형사
Adverb          부사
Conjunction     접속사
Exclamation     감탄사
Josa            조사
PreEomi         선어말 어미(eg.잇)
Eomi            어미
Suffix          접미사
Punctuation     구두점
Foreign         외국어
Alpha           알파벳
Number          숫자
Unknown         미등록어
KoreanParticle  (eg. ㅋㅋ, ㅎㅎ)
"""