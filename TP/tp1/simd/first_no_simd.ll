; ModuleID = 'first_no_simd.cpp'
source_filename = "first_no_simd.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%"struct.std::_Vector_base<double, std::allocator<double>>::_Vector_impl_data" = type { ptr, ptr, ptr }

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @_Z20memory_bound_no_simdRKSt6vectorIdSaIdEERS1_(ptr nocapture noundef nonnull readonly align 8 dereferenceable(24) %0, ptr nocapture noundef nonnull readonly align 8 dereferenceable(24) %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds %"struct.std::_Vector_base<double, std::allocator<double>>::_Vector_impl_data", ptr %0, i64 0, i32 1
  %4 = load ptr, ptr %3, align 8, !tbaa !5
  %5 = load ptr, ptr %0, align 8, !tbaa !10
  %6 = icmp eq ptr %4, %5
  br i1 %6, label %31, label %7

7:                                                ; preds = %2
  %8 = ptrtoint ptr %4 to i64
  %9 = ptrtoint ptr %5 to i64
  %10 = sub i64 %8, %9
  %11 = ashr exact i64 %10, 3
  %12 = load ptr, ptr %1, align 8, !tbaa !10
  %13 = tail call i64 @llvm.umax.i64(i64 %11, i64 1)
  %14 = and i64 %13, 7
  %15 = icmp ult i64 %11, 8
  br i1 %15, label %18, label %16

16:                                               ; preds = %7
  %17 = and i64 %13, -8
  br label %32

18:                                               ; preds = %32, %7
  %19 = phi i64 [ 0, %7 ], [ %74, %32 ]
  %20 = icmp eq i64 %14, 0
  br i1 %20, label %31, label %21

21:                                               ; preds = %18, %21
  %22 = phi i64 [ %28, %21 ], [ %19, %18 ]
  %23 = phi i64 [ %29, %21 ], [ 0, %18 ]
  %24 = getelementptr inbounds double, ptr %5, i64 %22
  %25 = load double, ptr %24, align 8, !tbaa !11
  %26 = fmul double %25, %25
  %27 = getelementptr inbounds double, ptr %12, i64 %22
  store double %26, ptr %27, align 8, !tbaa !11
  %28 = add nuw i64 %22, 1
  %29 = add i64 %23, 1
  %30 = icmp eq i64 %29, %14
  br i1 %30, label %31, label %21, !llvm.loop !13

31:                                               ; preds = %18, %21, %2
  ret void

32:                                               ; preds = %32, %16
  %33 = phi i64 [ 0, %16 ], [ %74, %32 ]
  %34 = phi i64 [ 0, %16 ], [ %75, %32 ]
  %35 = getelementptr inbounds double, ptr %5, i64 %33
  %36 = load double, ptr %35, align 8, !tbaa !11
  %37 = fmul double %36, %36
  %38 = getelementptr inbounds double, ptr %12, i64 %33
  store double %37, ptr %38, align 8, !tbaa !11
  %39 = or disjoint i64 %33, 1
  %40 = getelementptr inbounds double, ptr %5, i64 %39
  %41 = load double, ptr %40, align 8, !tbaa !11
  %42 = fmul double %41, %41
  %43 = getelementptr inbounds double, ptr %12, i64 %39
  store double %42, ptr %43, align 8, !tbaa !11
  %44 = or disjoint i64 %33, 2
  %45 = getelementptr inbounds double, ptr %5, i64 %44
  %46 = load double, ptr %45, align 8, !tbaa !11
  %47 = fmul double %46, %46
  %48 = getelementptr inbounds double, ptr %12, i64 %44
  store double %47, ptr %48, align 8, !tbaa !11
  %49 = or disjoint i64 %33, 3
  %50 = getelementptr inbounds double, ptr %5, i64 %49
  %51 = load double, ptr %50, align 8, !tbaa !11
  %52 = fmul double %51, %51
  %53 = getelementptr inbounds double, ptr %12, i64 %49
  store double %52, ptr %53, align 8, !tbaa !11
  %54 = or disjoint i64 %33, 4
  %55 = getelementptr inbounds double, ptr %5, i64 %54
  %56 = load double, ptr %55, align 8, !tbaa !11
  %57 = fmul double %56, %56
  %58 = getelementptr inbounds double, ptr %12, i64 %54
  store double %57, ptr %58, align 8, !tbaa !11
  %59 = or disjoint i64 %33, 5
  %60 = getelementptr inbounds double, ptr %5, i64 %59
  %61 = load double, ptr %60, align 8, !tbaa !11
  %62 = fmul double %61, %61
  %63 = getelementptr inbounds double, ptr %12, i64 %59
  store double %62, ptr %63, align 8, !tbaa !11
  %64 = or disjoint i64 %33, 6
  %65 = getelementptr inbounds double, ptr %5, i64 %64
  %66 = load double, ptr %65, align 8, !tbaa !11
  %67 = fmul double %66, %66
  %68 = getelementptr inbounds double, ptr %12, i64 %64
  store double %67, ptr %68, align 8, !tbaa !11
  %69 = or disjoint i64 %33, 7
  %70 = getelementptr inbounds double, ptr %5, i64 %69
  %71 = load double, ptr %70, align 8, !tbaa !11
  %72 = fmul double %71, %71
  %73 = getelementptr inbounds double, ptr %12, i64 %69
  store double %72, ptr %73, align 8, !tbaa !11
  %74 = add nuw i64 %33, 8
  %75 = add i64 %34, 8
  %76 = icmp eq i64 %75, %17
  br i1 %76, label %18, label %32, !llvm.loop !15
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="skylake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+sahf,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-complex,-amx-fp16,-amx-int8,-amx-tile,-avx10.1-256,-avx10.1-512,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512fp16,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,-cldemote,-clwb,-clzero,-cmpccxadd,-enqcmd,-fma4,-gfni,-hreset,-kl,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-pku,-prefetchi,-prefetchwt1,-ptwrite,-raoint,-rdpid,-rdpru,-rtm,-serialize,-sha,-sha512,-shstk,-sm3,-sm4,-sse4a,-tbm,-tsxldtrk,-uintr,-usermsr,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-widekl,-xop" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!5 = !{!6, !7, i64 8}
!6 = !{!"_ZTSNSt12_Vector_baseIdSaIdEE17_Vector_impl_dataE", !7, i64 0, !7, i64 8, !7, i64 16}
!7 = !{!"any pointer", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!6, !7, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !8, i64 0}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.unroll.disable"}
!15 = distinct !{!15, !16, !17}
!16 = !{!"llvm.loop.mustprogress"}
!17 = !{!"llvm.loop.vectorize.width", i32 1}
