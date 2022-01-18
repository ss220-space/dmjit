/proc/do_test()
    compile_proc(/proc/list_get)
    compile_proc(/proc/list_set)
    CHECK_INSTALL_COMPILED // RES: /list_get, /list_set
    RES(list_get(list(5,9,99),1)) // RES: 5
    RES(list_get(list(5,9,99),2)) // RES: 9
    RES(list_get(list(5,9,99),3)) // RES: 99
    var/list/test = list(0,0,0)
    list_set(test, 1, 6)
    RES(test[1]) // RES: 6
    RES(test[2]) // RES: 0
    RES(test[3]) // RES: 0
    list_set(test, 3, 99)
    RES(test[1]) // RES: 6
    RES(test[2]) // RES: 0
    RES(test[3]) // RES: 99
    test = list()
    test["test0"] = 770
    test["test1"] = 771
    test["test2"] = 772
    RES(list_get(test, "test0")) // RES: 770
    RES(list_get(test, "test1")) // RES: 771
    RES(list_get(test, "test2")) // RES: 772
    list_set(test, "test0", 990)
    list_set(test, "test1", 991)
    list_set(test, "test2", 992)
    RES(test["test0"]) // RES: 990
    RES(test["test1"]) // RES: 991
    RES(test["test2"]) // RES: 992
    dmjit_print_list_content(test)
    list_set(test, 1, 250)
    dmjit_print_list_content(test)
    RES(test["test0"]) // RES:
    RES(0) // RES: 0

/proc/list_get(a, b)
    return a[b]

/proc/list_set(a, b, c)
    a[b] = c